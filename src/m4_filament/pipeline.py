"""Train and apply the published M4 refiner on frozen E5 crops."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile

import numpy as np
from pycocotools import mask as coco
import torch

from m3_filament.io import read_jsonl, sha256, write_json, write_jsonl
from m3_filament.masks import decode
from random_validation.scoring import cc_record
from team_adaptation.association import prepare_components
from team_pipeline.contracts import load_probability

from .crop import CropTransform, make_input, plan_crop
from .data import CHECKPOINTS, PROTOCOL, load_json, probability_pair, read_gray, unique_records
from .recipes import (
    RULES, SIZES, TRIALS, augment, learning_rate, make_model,
    objective, sample_weight, shuffled_batches,
)


def native_mask(full):
    rle = coco.encode(np.asfortranarray(full,dtype=np.uint8))
    return {"size":list(rle["size"]),"counts":rle["counts"].decode("ascii")}


def paste_slice(crop):
    x,y = crop.origin_xy
    h,w = crop.original_hw
    ch,cw = crop.crop_hw
    x0,y0,x1,y1 = max(0,x),max(0,y),min(w,x+cw),min(h,y+ch)
    return (slice(y0,y1),slice(x0,x1)),(slice(y0-y,y1-y),slice(x0-x,x1-x))


def batch_probabilities(model, arrays, instances, device):
    pending = defaultdict(list)
    for i,row in enumerate(instances):
        if row["array"] is not None:
            x = arrays[row["array"]]
            if x.shape != (3, *row["transform"]["crop_hw"]) or x.dtype != np.float32:
                raise ValueError("Invalid inference crop")
            pending[x.shape[-1]].append((i,x))
    result = {}
    for side,group in pending.items():
        size = SIZES[side]
        for start in range(0,len(group),size):
            block = group[start:start+size]
            x = torch.from_numpy(np.stack([x for _,x in block])).to(device)
            p = model(x).sigmoid().cpu().numpy()[:,0]
            if not np.isfinite(p).all():
                raise ValueError("Nonfinite predicted mask")
            for (i,array),probability in zip(block,p):
                result[i]=(probability,array[2])
    return result


def apply_rules(instances, probabilities, shape, rules):
    records = {rule:[] for rule in rules}
    used = {rule:np.zeros(shape,dtype=bool) for rule in rules}
    canvas = np.zeros(shape,dtype=np.uint8,order="F")
    for i,row in enumerate(instances):
        if i in probabilities:
            source,local = paste_slice(CropTransform.from_dict(row["transform"]))
            probability,coarse = probabilities[i]
        else:
            source = local = np.s_[:,:]
            original = decode(row["segmentation"]).astype(bool)
        for rule,(threshold,weight) in rules.items():
            if i in probabilities:
                values = probability if weight==0 else (1-weight)*probability+weight*coarse
                mask = values[local]>=threshold
            else:
                mask = original
            region = used[rule][source]
            keep = mask & ~region
            if keep.any():
                canvas[source] = keep
                rle = native_mask(canvas)
                canvas[source] = 0
                region |= keep
                records[rule].append({"instance_id":row["instance_id"],"score":row["score"],"segmentation":rle})
    return records



def save_weights(path, checkpoint):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".m4-save-") as temporary:
        staged = Path(temporary) / path.name
        torch.save(checkpoint, staged)
        staged.replace(path)


def train(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        if not torch.cuda.is_bf16_supported():
            raise ValueError("The published training protocol requires CUDA BF16")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    dataset = Path(args.dataset)
    report = load_json(dataset / "report.json")
    manifest = dataset / "manifest.jsonl"
    if sha256(manifest) != report["manifest_sha256"] or sha256(PROTOCOL) != report["protocol_sha256"]:
        raise ValueError("Dataset manifest or protocol differs")
    protocol = load_json(PROTOCOL)["observations"]
    allowed = {k for k,v in protocol.items() if v["split"] == "train"}
    if args.exclude:
        excluded = set(load_json(args.exclude))
        if not excluded <= allowed:
            raise ValueError("Development exclusion includes observations outside the training partition")
        groups = {protocol[o]["group"] for o in excluded}
        allowed = {o for o in allowed if protocol[o]["group"] not in groups}
    rows = list(read_jsonl(manifest))
    if any(row["kind"] != "matched" or protocol[row["observation_id"]]["split"] != "train" for row in rows):
        raise ValueError("Training requires matched positives from the original training partition")
    rows = [row for row in rows if row["observation_id"] in allowed]
    if not rows:
        raise ValueError("No training samples after date-group exclusion")
    records = [(row["sample_id"], None, row) for row in rows]
    batches = list(shuffled_batches(records, args.seed))
    steps = len(batches) * args.stop_after // 4
    buckets = defaultdict(list)
    coordinates = []
    for row in rows:
        path = (dataset / row["npz"]).resolve()
        if path.parent != (dataset / "samples").resolve() or sha256(path) != row["npz_sha256"]:
            raise ValueError("Crop file differs from its manifest")
        with np.load(path, allow_pickle=False) as sample:
            x, y = sample["X"], sample["Y"]
        size = row["transform"]["crop_hw"][0]
        if x.shape != (3, size, size) or y.shape != (1, size, size) or x.dtype != np.float32 or y.dtype != np.float32:
            raise ValueError("Invalid crop tensor")
        coordinates.append(len(buckets[size]))
        buckets[size].append(np.concatenate((x,y),axis=0))
    cache = {size:torch.from_numpy(np.stack(samples)).to(device) for size,samples in buckets.items()}
    del buckets
    config = TRIALS[args.recipe]["config"]
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    counts = Counter(row["observation_id"] for row in rows)
    for step, (size, indices) in enumerate(batches[:steps]):
        block = cache[size][[coordinates[i] for i in indices]]
        x, y = augment(block[:,:3], block[:,3:], config, args.seed + step)
        weights = sample_weight(config, [rows[i] for i in indices], y, counts)
        optimizer.param_groups[0]["lr"] = learning_rate(config, step, len(batches))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        loss = objective(logits, y, x[:,2:3], config, weights)
        if not torch.isfinite(loss):
            raise ValueError("M4 training loss diverged")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 200 == 0 or step + 1 == steps:
            print(json.dumps({"step":step+1,"steps":steps,"loss":float(loss.detach())}), flush=True)
    save_weights(args.output, {"format":"solar-filament-m4-v1","recipe":args.recipe,
                               "config":config,"threshold":args.threshold,"model_state":model.state_dict()})
    info = {"recipe":args.recipe,"epochs":args.stop_after,"samples":len(rows),"seed":args.seed,
            "steps":steps,"manifest_sha256":sha256(manifest),"checkpoint_sha256":sha256(args.output),
            "precision":"cuda_bf16_forward" if device.type == "cuda" else "cpu_float32"}
    print(json.dumps(info))


def refine(args):
    torch.set_num_threads(4)
    cv_records = unique_records(args.coarse)
    weights = load_json(CHECKPOINTS)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "solar-filament-m4-v1":
        raise ValueError("Expected an exported M4 checkpoint")
    model = make_model(checkpoint["config"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    policy = RULES[args.rule] if args.rule else (checkpoint["threshold"],0.)
    output = []
    with torch.inference_mode():
        for obs, record in cv_records.items():
            image, gray = read_gray(args.images, obs)
            if tuple(record["image_size"]) != gray.shape:
                raise ValueError(f"Original coordinates differ for {obs}")
            probability = probability_pair(args.m1, args.m2, obs, gray.shape, sha256(image), weights)
            instances, arrays = [], {}
            ordered = sorted(record["instances"], key=lambda r:(-r["score"], r["instance_id"]))
            for i,row in enumerate(ordered):
                mask = decode(row["segmentation"]).astype(bool)
                try:
                    crop = plan_crop(mask, context_px=32)
                except ValueError:
                    instances.append({**row,"array":None,"transform":None})
                    continue
                key = str(i)
                arrays[key] = make_input(gray, probability, mask, crop)
                instances.append({**row,"array":key,"transform":crop.to_dict()})
            values = batch_probabilities(model, arrays, instances, device)
            predicted = apply_rules(instances, values, gray.shape, {"m4":policy})["m4"]
            output.append({"observation_id":obs,"image_size":list(gray.shape),"instances":predicted})
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    write_jsonl(path, output)
    print(json.dumps({"observations":len(output),"rule":list(policy),"output_sha256":sha256(path),
                      "coarse_sha256":sha256(args.coarse),"checkpoint_sha256":sha256(args.checkpoint)}))


def components(args):
    threshold = .50 if args.model == "m1" else .35
    key = "team/m1/weights/best_model.pth" if args.model == "m1" else "team/m2/weights/best.pt"
    expected = load_json(CHECKPOINTS)[key]
    records = []
    for path in sorted(Path(args.probabilities).glob("*.npz")):
        obs = path.stem
        image, gray = read_gray(args.images, obs)
        values, meta = load_probability(path, obs, gray.shape)
        if (meta["model"],meta["checkpoint_hash"],meta["image_sha256"]) != (args.model,expected,sha256(image)):
            raise ValueError(f"Semantic cache provenance differs for {obs}")
        prepared = prepare_components(values, threshold, 400)
        records.append(cc_record(obs, prepared, {args.model:meta}, args.model.upper()+"_CC"))
    if not records:
        raise ValueError("No cached probabilities")
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    write_jsonl(path, records)
    print(json.dumps({"observations":len(records),"threshold":threshold,"output_sha256":sha256(path)}))
