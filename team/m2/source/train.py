#!/usr/bin/env python
"""M2 - SegFormer semantic segmentation for the Solar Filament Segmentation Challenge 2026.

Trains a SegFormer (HuggingFace transformers) on full-resolution crops and outputs the
per-pixel filament probability P_T in original 2048x2048 image coordinates (same
interface as M1's CNN).

This file also holds the helpers shared with test.py: annotation loading, the
observation-grouped split, the model wrapper, sliding-window inference, and the
team evaluator that mirrors the organisers' Self_Evaluation_Notebook (v6).

Usage (see README.md):
    python train.py --run-name b2_base
    python train.py --run-name b2_oof --fold 0          # train without inner fold 0 (OOF)
    python train.py --limit 8 --epochs 2 --out-dir temp/runs   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util

ROOT = Path(__file__).resolve().parent
DEFAULT_ANN = ROOT / "data/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMG_DIR = ROOT / "data/train/train_images"
TEST_IMG_DIR = ROOT / "data/test/test_images"
H = W = 2048
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Official Self_Evaluation_Notebook version that the evaluator below mirrors.
OFFICIAL_EVAL = {
    "ref": "azimahmadzadeh/self-evaluation-notebook",
    "version": 6,
    "source_sha256": "ae7c9445b4eee2b77b216df5c9772fdf92fda15611e98abd7ef5f02a4e380801",
    "iou_threshold": 0.5,  # strict: IoU > 0.5
}

# Fixed error-slice definitions (do not change between runs, or comparisons break).
SMALL_AREA = 1000      # px, ~41% of training filaments are below this
LARGE_AREA = 5000      # px
SLENDER_ELONG = 16.0   # spine_length^2 / area, ~top quartile of training filaments
ADJACENT_GAP = 20      # px, bbox gap to another filament of the same annotator record


# --------------------------------------------------------------------------- utils
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def env_info():
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "opencv": cv2.__version__,
            "cuda": torch.version.cuda, "platform": platform.platform()}
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


# --------------------------------------------------------------------------- data
def load_annotations(ann_path=DEFAULT_ANN):
    """Group the COCO json by observation.

    Returns {stem: {"file": file_name, "records": {record_id: [ann, ...]}}}.
    One observation (image file) can have several annotator records; each record
    id looks like "050101-20111116063134Lh" and is the official annotator-image key.
    """
    raw = json.loads(Path(ann_path).read_text())
    obs, rec2stem = {}, {}
    for im in raw["images"]:
        if (im["height"], im["width"]) != (H, W):
            raise ValueError(f"unexpected size for {im['id']}: {im['height']}x{im['width']}")
        stem = Path(im["file_name"]).stem
        obs.setdefault(stem, {"file": im["file_name"], "records": {}})["records"][im["id"]] = []
        rec2stem[im["id"]] = stem
    for a in raw["annotations"]:
        obs[rec2stem[a["image_id"]]]["records"][a["image_id"]].append(a)
    return obs


def polygon_rle(ann):
    """COCO polygon(s) of one filament -> one RLE (pycocotools rasterisation)."""
    return mask_util.merge(mask_util.frPyObjects(ann["segmentation"], H, W))


def semantic_label(o):
    """Soft label = mean over annotator records of each record's instance union.

    Instances of different annotators are never mixed into a new instance set;
    only the semantic (foreground) maps are averaged.
    """
    acc = np.zeros((H, W), np.float32)
    for anns in o["records"].values():
        if anns:
            acc += mask_util.decode(mask_util.merge([polygon_rle(a) for a in anns]))
    return acc / len(o["records"])


def label_cache_dir(ann_path):
    return ROOT / "cache" / f"labels_mean_{sha256_file(ann_path)[:12]}"


def build_label_cache(obs, stems, ann_path):
    """Write uint8 soft labels (round(255 * mean)) as PNG once; returns the cache dir."""
    out = label_cache_dir(ann_path)
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in stems if not (out / f"{s}.png").exists()]
    for i, s in enumerate(todo):
        lab = np.round(semantic_label(obs[s]) * 255).astype(np.uint8)
        cv2.imwrite(str(out / f"{s}.png"), lab)
        if (i + 1) % 100 == 0:
            print(f"  label cache {i + 1}/{len(todo)}", flush=True)
    return out


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape != (H, W):
        raise ValueError(f"{path}: expected {H}x{W}, got {img.shape}")
    return img


# --------------------------------------------------------------------------- split
def make_split(obs, seed=0, ratios=(0.70, 0.15, 0.15), n_folds=5):
    """Split observations by observation date (YYYYMMDD) into train/calib/holdout.

    All annotator records of one file share a stem, and all stems of the same day share
    a group, so duplicates and same-day near-duplicates never cross sets. Train groups
    are further assigned to n_folds inner folds for out-of-fold (OOF) predictions.
    """
    groups = sorted({s[:8] for s in obs})
    random.Random(seed).shuffle(groups)
    n_tr = round(len(groups) * ratios[0])
    n_ca = round(len(groups) * ratios[1])
    split_of = {g: "train" for g in groups[:n_tr]}
    split_of.update({g: "calib" for g in groups[n_tr:n_tr + n_ca]})
    split_of.update({g: "holdout" for g in groups[n_tr + n_ca:]})
    fold_of = {g: i % n_folds for i, g in enumerate(groups[:n_tr])}
    observations = {s: {"group": s[:8], "split": split_of[s[:8]], "fold": fold_of.get(s[:8]),
                        "records": sorted(obs[s]["records"])} for s in sorted(obs)}
    return {"group_by": "date(YYYYMMDD)", "seed": seed, "ratios": list(ratios),
            "n_folds": n_folds, "observations": observations,
            "split_hash": sha256_json(observations)}


def load_split(path):
    split = json.loads(Path(path).read_text())
    if sha256_json(split["observations"]) != split["split_hash"]:
        raise ValueError(f"{path}: split_hash does not match its observations")
    return split


def subset_stems(split, subset, fold=None):
    """Stems of a subset. 'train' with fold=k excludes inner fold k; 'oof' is fold k only."""
    o = split["observations"]
    if subset == "train":
        return sorted(s for s, v in o.items() if v["split"] == "train" and (fold is None or v["fold"] != fold))
    if subset == "oof":
        if fold is None:
            raise ValueError("subset 'oof' needs --fold")
        return sorted(s for s, v in o.items() if v["split"] == "train" and v["fold"] == fold)
    if subset in ("calib", "holdout"):
        return sorted(s for s, v in o.items() if v["split"] == subset)
    raise ValueError(f"unknown subset {subset}")


class CropDataset(torch.utils.data.Dataset):
    """Random full-resolution crops; a fraction is centred near filament pixels."""

    def __init__(self, obs, stems, label_dir, crop, crops_per_image, pos_frac, seed):
        self.files = [TRAIN_IMG_DIR / obs[s]["file"] for s in stems]
        self.labels = [label_dir / f"{s}.png" for s in stems]
        self.crop, self.pos_frac, self.seed = crop, pos_frac, seed
        self.n = len(stems) * crops_per_image
        self.epoch = 0
        rng = np.random.default_rng(seed)
        self.fg = []  # sub-sampled foreground coordinates for positive crops
        for p in self.labels:
            ys, xs = np.nonzero(cv2.imread(str(p), cv2.IMREAD_UNCHANGED))
            idx = rng.choice(len(ys), size=min(len(ys), 4096), replace=False) if len(ys) else []
            self.fg.append(np.stack([ys[idx], xs[idx]], 1).astype(np.int32))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        k = i % len(self.files)
        img, lab = read_gray(self.files[k]), cv2.imread(str(self.labels[k]), cv2.IMREAD_UNCHANGED)
        c, fg = self.crop, self.fg[k]
        if len(fg) and rng.random() < self.pos_frac:
            cy, cx = fg[rng.integers(len(fg))] + rng.integers(-c // 4, c // 4 + 1, size=2)
            y0, x0 = int(np.clip(cy - c // 2, 0, H - c)), int(np.clip(cx - c // 2, 0, W - c))
        else:
            y0, x0 = (int(v) for v in rng.integers(0, H - c + 1, size=2))
        img = img[y0:y0 + c, x0:x0 + c].astype(np.float32) / 255.0
        lab = lab[y0:y0 + c, x0:x0 + c].astype(np.float32) / 255.0
        # basic augmentation: dihedral transforms + mild linear intensity jitter
        r = int(rng.integers(4))
        img, lab = np.rot90(img, r), np.rot90(lab, r)
        if rng.random() < 0.5:
            img, lab = img[:, ::-1], lab[:, ::-1]
        img = np.clip(img * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05), 0, 1)
        return (torch.from_numpy(np.ascontiguousarray(img))[None],
                torch.from_numpy(np.ascontiguousarray(lab))[None])


# --------------------------------------------------------------------------- model
def build_model(backbone=None, hf_config=None):
    """Pretrained encoder from the hub (training), or architecture-only from a saved config (inference)."""
    # nvidia/mit-b* only ship pytorch_model.bin; without this, transformers starts a non-daemon thread
    # that tries a safetensors conversion on the Hub and can block process exit for minutes.
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    if hf_config is not None:
        return SegformerForSemanticSegmentation(SegformerConfig.from_dict(hf_config))
    try:  # avoid a slow hub round-trip when the weights are already cached
        return SegformerForSemanticSegmentation.from_pretrained(backbone, num_labels=1, local_files_only=True)
    except OSError:
        return SegformerForSemanticSegmentation.from_pretrained(backbone, num_labels=1)


def forward_logits(model, x):
    """x: (B,1,h,w) in [0,1] -> logits (B,1,h,w). Grey is replicated to 3 channels."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    logits = model(pixel_values=(x.expand(-1, 3, -1, -1) - mean) / std).logits  # 1/4 resolution
    return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def autocast(device):
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16)


@torch.no_grad()
def predict_prob(model, img_u8, device, tile=512, overlap=128, batch=4):
    """Sliding-window inference at native resolution.

    Overlapping tiles are blended in probability space with a Hann window, so the
    result is directly an HxW probability map in original image coordinates.
    """
    h, w = img_u8.shape
    stride = tile - overlap
    ys = sorted(set(list(range(0, h - tile + 1, stride)) + [h - tile]))
    xs = sorted(set(list(range(0, w - tile + 1, stride)) + [w - tile]))
    win1d = torch.hann_window(tile + 2, periodic=False, device=device)[1:-1]
    win = (win1d[:, None] * win1d[None, :])[None]
    img = torch.from_numpy(img_u8).to(device).float().div_(255.0)
    acc = torch.zeros((h, w), device=device)
    wsum = torch.zeros((h, w), device=device)
    coords = [(y, x) for y in ys for x in xs]
    for i in range(0, len(coords), batch):
        cs = coords[i:i + batch]
        x = torch.stack([img[y:y + tile, x0:x0 + tile] for y, x0 in cs])[:, None]
        with autocast(device):
            p = torch.sigmoid(forward_logits(model, x).float())[:, 0]
        for (y, x0), pi in zip(cs, p):
            acc[y:y + tile, x0:x0 + tile] += pi * win[0]
            wsum[y:y + tile, x0:x0 + tile] += win[0]
    return (acc / wsum).cpu().numpy()


def loss_fn(logits, target, lambda_dice):
    """L = BCE + lambda_D * soft Dice (batch-level)."""
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
    return bce + lambda_dice * dice, bce, dice


# --------------------------------------------------------------------------- evaluator
# Mirrors the official notebook: for every annotator record (annotator-image) the
# record's GT filaments are matched against ALL predictions of that image; a pair is a
# hit when IoU > 0.5 (every hit pair counts as TP); FP = predictions without a hit,
# FN = GT without a hit; n_gt == 0 -> all FP, n_pred == 0 -> all FN; TP/FP/FN and the
# IoU sum are pooled over all records, PQ = sum IoU(TP) / (TP + 0.5 FP + 0.5 FN).
def official_counts(iou):
    n_gt, n_pred = iou.shape
    if n_gt == 0:
        return {"tp": 0, "fp": n_pred, "fn": 0, "iou_sum": 0.0}
    if n_pred == 0:
        return {"tp": 0, "fp": 0, "fn": n_gt, "iou_sum": 0.0}
    hit = iou > OFFICIAL_EVAL["iou_threshold"]
    return {"tp": int(hit.sum()), "fp": int((hit.sum(0) == 0).sum()),
            "fn": int((hit.sum(1) == 0).sum()), "iou_sum": float(iou[hit].sum())}


def add_counts(acc, c):
    for k in ("tp", "fp", "fn", "iou_sum"):
        acc[k] = acc.get(k, 0) + c[k]
    return acc


def pq_summary(c):
    tp, fp, fn, s = c.get("tp", 0), c.get("fp", 0), c.get("fn", 0), c.get("iou_sum", 0.0)
    den = tp + 0.5 * fp + 0.5 * fn
    return {"PQ": s / den if den else 0.0, "SQ": s / tp if tp else 0.0,
            "RQ": tp / den if den else 0.0, "tp": tp, "fp": fp, "fn": fn, "iou_sum": s}


def gt_for_stem(o):
    """{record_id: [instance]} with flat pixel indices (row-major) and fixed slice flags.

    Records without any filament are skipped, as in the official code (they have no GT rows).
    """
    recs = {}
    for rid, anns in sorted(o["records"].items()):
        if not anns:
            continue
        insts = []
        for a in anns:
            idx = np.flatnonzero(mask_util.decode(polygon_rle(a)))
            spine = np.asarray(a.get("spine") or [], np.float64).reshape(-1, 2)
            length = np.linalg.norm(np.diff(spine, axis=0), axis=1).sum() if len(spine) > 1 else 0.0
            insts.append({"idx": idx, "area": len(idx), "bbox": a["bbox"],
                          "small": len(idx) < SMALL_AREA, "large": len(idx) >= LARGE_AREA,
                          "slender": length ** 2 / max(a["area"], 1) >= SLENDER_ELONG})
        for i, g in enumerate(insts):  # adjacency: another filament's bbox within ADJACENT_GAP px
            x, y, w, h = g["bbox"]
            g["adjacent"] = any(max(o2["bbox"][0] - (x + w), x - (o2["bbox"][0] + o2["bbox"][2]),
                                    o2["bbox"][1] - (y + h), y - (o2["bbox"][1] + o2["bbox"][3])) < ADJACENT_GAP
                                for j, o2 in enumerate(insts) if j != i)
        recs[rid] = insts
    return recs


def components(prob, thr):
    """Fixed basic instantiation: threshold (>= thr) + 8-connected components."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats((prob >= thr).astype(np.uint8), connectivity=8)
    return lab.ravel(), stats[1:, cv2.CC_STAT_AREA].astype(np.int64)


def iou_labelmap(insts, lab_flat, areas):
    """IoU (n_gt, n_pred) between GT instances and non-overlapping predicted components."""
    m = np.zeros((len(insts), len(areas)))
    for i, g in enumerate(insts):
        inter = np.bincount(lab_flat[g["idx"]], minlength=len(areas) + 1)[1:]
        m[i] = inter / (g["area"] + areas - inter)
    return m


def score_grid(prob, gt_recs, thrs, min_areas):
    """Pooled counts of one observation for every (thr, min_area) of the grid."""
    out = {}
    for thr in thrs:
        lab, areas = components(prob, thr)
        ious = [iou_labelmap(insts, lab, areas) for insts in gt_recs.values()]
        for ma in min_areas:
            keep = areas >= ma
            acc = {}
            for m in ious:
                add_counts(acc, official_counts(m[:, keep]))
            out[(thr, ma)] = acc
    return out


def score_detail(prob, gt_recs, thr, min_area):
    """Counts plus error-analysis detail for one observation at a fixed (thr, min_area)."""
    lab, areas = components(prob, thr)
    keep = areas >= min_area
    pred_fg = np.concatenate([[False], keep])[lab]
    d = {"counts": {}, "gt_hits": [], "fp_areas": [], "gt_split": 0, "pred_merge": 0,
         "dice_inter": 0, "dice_pred": 0, "dice_gt": 0, "n_pred": int(keep.sum())}
    for rid, insts in gt_recs.items():
        m = iou_labelmap(insts, lab, areas)[:, keep]
        add_counts(d["counts"], official_counts(m))
        hit = m > OFFICIAL_EVAL["iou_threshold"]
        touch = m > 0
        d["gt_split"] += int((touch.sum(1) >= 2).sum())   # GT overlapped by >=2 predictions
        d["pred_merge"] += int((touch.sum(0) >= 2).sum())  # prediction overlapping >=2 GT
        d["fp_areas"] += areas[keep][hit.sum(0) == 0].tolist()
        d["gt_hits"] += [{"record": rid, "k": i, "hit": bool(hit[i].any()),
                          **{s: g[s] for s in ("area", "small", "large", "slender", "adjacent")}}
                         for i, g in enumerate(insts)]
        union = np.unique(np.concatenate([g["idx"] for g in insts])) if insts else np.zeros(0, np.int64)
        d["dice_inter"] += int(pred_fg[union].sum())
        d["dice_pred"] += int(pred_fg.sum())
        d["dice_gt"] += len(union)
    return d


def evaluate_probs(obs, stems, get_prob, thr, min_area):
    """Fixed-parameter evaluation over many observations (used for training validation)."""
    acc = {}
    for s in stems:
        add_counts(acc, score_grid(get_prob(s), gt_for_stem(obs[s]), [thr], [min_area])[(thr, min_area)])
    return pq_summary(acc)


# --------------------------------------------------------------------------- npz cache
class NpzWriter:
    """Streams arrays into a .npz (zip) one by one, so a whole subset never sits in RAM."""

    def __init__(self, path):
        self.zf = zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED, allowZip64=True)

    def add(self, key, arr):
        with self.zf.open(f"{key}.npy", "w", force_zip64=True) as f:
            np.lib.format.write_array(f, np.asanyarray(arr), allow_pickle=False)

    def close(self):
        self.zf.close()


# --------------------------------------------------------------------------- training
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ann", default=str(DEFAULT_ANN))
    p.add_argument("--split", default=None, help="existing split_manifest.json (e.g. from M1); default: make one")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--fold", type=int, default=None, help="train without inner fold k (for OOF probabilities)")
    p.add_argument("--backbone", default="nvidia/mit-b2", help="HF encoder weights, e.g. nvidia/mit-b0..b5")
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--crops-per-image", type=int, default=4, help="crops per training image per epoch")
    p.add_argument("--pos-frac", type=float, default=0.5, help="fraction of crops centred near filaments")
    p.add_argument("--lr", type=float, default=6e-5, help="encoder lr; decode head uses 10x")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--lambda-dice", type=float, default=1.0)
    p.add_argument("--val-every", type=int, default=5, help="epochs between calibration evaluations")
    p.add_argument("--val-thr", type=float, default=0.5, help="fixed monitoring threshold")
    p.add_argument("--val-min-area", type=int, default=200, help="fixed monitoring min component area")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="use only N train / N calib observations (smoke test)")
    p.add_argument("--out-dir", default=str(ROOT / "runs"))
    p.add_argument("--run-name", default="segformer")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = f"{args.run_name}{'' if args.fold is None else f'_fold{args.fold}'}_{datetime.now():%Y%m%d_%H%M%S}"
    out = Path(args.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=False)
    print(f"run_id={run_id}  device={device}", flush=True)

    obs = load_annotations(args.ann)
    split = load_split(args.split) if args.split else make_split(obs, seed=args.split_seed)
    (out / "split_manifest.json").write_text(json.dumps(split, indent=1))
    if args.fold is not None and not 0 <= args.fold < split["n_folds"]:
        raise ValueError(f"--fold must be in [0, {split['n_folds']})")
    train_stems = subset_stems(split, "train", args.fold)
    calib_stems = subset_stems(split, "calib")
    if args.limit:
        train_stems, calib_stems = train_stems[:args.limit], calib_stems[:args.limit]
    print(f"split_hash={split['split_hash'][:12]}  train={len(train_stems)}  calib={len(calib_stems)}", flush=True)

    label_dir = build_label_cache(obs, train_stems, args.ann)
    ds = CropDataset(obs, train_stems, label_dir, args.crop, args.crops_per_image, args.pos_frac, args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    model = build_model(args.backbone).to(device)
    head = [p for n, p in model.named_parameters() if n.startswith("decode_head")]
    enc = [p for n, p in model.named_parameters() if not n.startswith("decode_head")]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr}, {"params": head, "lr": args.lr * 10}],
                            weight_decay=args.weight_decay)
    steps_per_epoch = len(ds) // args.batch_size
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup_steps) * max(0.0, 1 - s / total))

    manifest = {"run_id": run_id, "task": "M2 SegFormer semantic segmentation (E2)",
                "code_sha256": sha256_file(__file__), "annotation_sha256": sha256_file(args.ann),
                "split_hash": split["split_hash"], "seed": args.seed, "args": vars(args), "env": env_info(),
                "annotation_policy": "soft label = mean over annotator records of each record's instance union; "
                                     "polygons rasterised with pycocotools",
                "n_train_obs": len(train_stems), "n_calib_obs": len(calib_stems),
                "evaluator": OFFICIAL_EVAL, "train_stems_sha256": sha256_json(train_stems)}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=1))

    log_f = open(out / "train_log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["epoch", "step", "loss", "bce", "dice", "lr", "calib_PQ", "calib_SQ", "calib_RQ",
                  "tp", "fp", "fn", "epoch_sec"])
    best_pq, step, t_start = -1.0, 0, time.time()

    def save(path, epoch, metrics):
        torch.save({"model": model.state_dict(), "hf_config": model.config.to_dict(), "args": vars(args),
                    "run_id": run_id, "epoch": epoch, "step": step, "split_hash": split["split_hash"],
                    "fold": args.fold, "calib_metrics": metrics}, path)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        ds.epoch = epoch
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                         num_workers=args.num_workers, pin_memory=True, generator=gen)
        model.train()
        sums = np.zeros(3)
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device):
                logits = forward_logits(model, x)
            loss, bce, dice = loss_fn(logits, y, args.lambda_dice)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            sums += [loss.item(), bce.item(), dice.item()]
        sums /= max(1, steps_per_epoch)

        metrics = None
        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            torch.cuda.empty_cache()
            metrics = evaluate_probs(obs, calib_stems,
                                     lambda s: predict_prob(model, read_gray(TRAIN_IMG_DIR / obs[s]["file"]), device),
                                     args.val_thr, args.val_min_area)
            if metrics["PQ"] > best_pq:
                best_pq = metrics["PQ"]
                save(out / "best.pt", epoch, metrics)
        save(out / "last.pt", epoch, metrics)
        m = metrics or {}
        row = [epoch, step, *np.round(sums, 5), f"{sched.get_last_lr()[0]:.2e}",
               *(round(m[k], 5) if k in m else "" for k in ("PQ", "SQ", "RQ")),
               *(m.get(k, "") for k in ("tp", "fp", "fn")), round(time.time() - t0, 1)]
        log.writerow(row)
        log_f.flush()
        print(f"epoch {epoch}/{args.epochs} step {step} loss {sums[0]:.4f} (bce {sums[1]:.4f} dice {sums[2]:.4f})"
              + (f" | calib PQ {m['PQ']:.4f} SQ {m['SQ']:.4f} RQ {m['RQ']:.4f} TP {m['tp']} FP {m['fp']} FN {m['fn']}"
                 if metrics else "") + f" | {time.time() - t0:.0f}s", flush=True)
    log_f.close()

    hours = (time.time() - t_start) / 3600
    manifest.update({"optimizer_steps": step, "samples_seen": step * args.batch_size, "gpu_hours": round(hours, 3),
                     "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
                     "best_calib_PQ": best_pq,
                     "checkpoints": {p.name: sha256_file(p) for p in sorted(out.glob("*.pt"))}})
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done: {out}  best calib PQ (thr={args.val_thr}, min_area={args.val_min_area}) = {best_pq:.4f}  "
          f"GPU hours = {hours:.2f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
