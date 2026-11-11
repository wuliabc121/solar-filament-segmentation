"""Build positive training crops and score native-coordinate predictions."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
from pycocotools import mask as coco

from m3_filament.evaluation import aggregate_counts, pairwise_iou, pq_counts
from m3_filament.io import read_jsonl, sha256, write_json, write_jsonl
from m3_filament.masks import decode, encode
from team_pipeline.contracts import load_probability

from .crop import crop_array, make_input, paste_array, plan_crop

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "team/protocols/m2_original.json"
CHECKPOINTS = ROOT / "docs/checkpoints.json"
RECIPE_ORDER = ROOT / "experiments/m4_e5/train_order.json"


def load_json(path):
    return json.loads(Path(path).read_text())


def unique_records(path):
    records = {}
    for record in read_jsonl(path):
        obs = record["observation_id"]
        if obs in records or Path(obs).name != obs or obs in ("", ".", ".."):
            raise ValueError(f"Invalid or duplicate observation ID: {obs}")
        records[obs] = record
    if not records:
        raise ValueError(f"No predictions in {path}")
    return records


def read_gray(images, observation):
    if Path(observation).name != observation or observation in ("", ".", ".."):
        raise ValueError(f"Unsafe observation ID: {observation}")
    path = Path(images) / f"{observation}.jpeg"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unreadable input image: {path}")
    return path, image


def probability_pair(m1_dir, m2_dir, observation, shape, image_hash, weights):
    values = []
    for model, directory, weight_key in (
        ("m1", m1_dir, "team/m1/weights/best_model.pth"),
        ("m2", m2_dir, "team/m2/weights/best.pt"),
    ):
        probability, info = load_probability(Path(directory) / f"{observation}.npz", observation, shape)
        if (info["model"], info["checkpoint_hash"], info["image_sha256"], info["purpose"]) != (
            model, weights[weight_key], image_hash, "application"
        ) or info["is_oof"] is not False:
            raise ValueError(f"{model} cache provenance differs for {observation}")
        values.append(probability)
    return .5 * values[0] + .5 * values[1]


def annotation_mask(row, shape):
    segmentation = row["segmentation"]
    if isinstance(segmentation, list):
        segmentation = coco.merge(coco.frPyObjects(segmentation, *shape))
    elif isinstance(segmentation["counts"], list):
        segmentation = coco.frPyObjects(segmentation, *shape)
    mask = coco.decode(segmentation).astype(bool)
    if mask.shape != tuple(shape):
        raise ValueError("GT mask shape differs from the original image")
    return mask


def annotation_index(path, records, protocol):
    annotations = load_json(path)
    by_image = defaultdict(list)
    expected_ids = {str(record_id): observation for observation in records
                    for record_id in protocol[observation]["records"]}
    images = {}
    for image in annotations["images"]:
        key = str(image["id"])
        if key in expected_ids:
            observation = Path(image["file_name"]).stem
            if observation != expected_ids[key] or key in images or [image["height"], image["width"]] != records[observation]["image_size"]:
                raise ValueError(f"Annotation dimensions or ID differ for {key}")
            images[key] = observation
    if set(images) != set(expected_ids):
        raise ValueError("Annotation records do not cover the requested protocol")
    for row in annotations["annotations"]:
        key = str(row["image_id"])
        if key in images:
            by_image[key].append(row)
    return images, by_image


def best_match(matrix, column):
    """Match the candidate to its unique mutual best GT with the published margin."""
    if not matrix.shape[0]:
        return None
    overlaps = matrix[:, column]
    gt = int(overlaps.argmax())
    best = float(overlaps[gt])
    second = float(np.sort(overlaps)[-2]) if len(overlaps) > 1 else 0.
    row = matrix[gt]
    if best < .30 or best - second < .05 or int(row.argmax()) != column or (row == best).sum() != 1:
        return None
    return gt, best


def prepare(args):
    protocol = load_json(PROTOCOL)["observations"]
    records = unique_records(args.coarse)
    train_ids = {key for key, value in protocol.items() if value["split"] == "train"}
    if set(records) != train_ids:
        raise ValueError("Preparation expects E5 candidates for the complete 500-image training partition")
    order = load_json(RECIPE_ORDER)
    if len(order) != len(train_ids) or set(order) != train_ids:
        raise ValueError("Training observation order differs from the frozen plan")
    _, gt_by_id = annotation_index(args.annotations, records, protocol)
    weights = load_json(CHECKPOINTS)
    rows, counts = [], Counter()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".m4-prepare-") as temporary:
        stage = Path(temporary) / "data"
        (stage / "samples").mkdir(parents=True)
        for obs in order:
            record = records[obs]
            image_path, gray = read_gray(args.images, obs)
            if tuple(record["image_size"]) != gray.shape:
                raise ValueError(f"Image coordinates differ for {obs}")
            image_hash = sha256(image_path)
            probability = probability_pair(args.m1, args.m2, obs, gray.shape, image_hash, weights)
            reference = sorted(map(str, protocol[obs]["records"]))[0]
            labels = sorted(gt_by_id[reference], key=lambda row: str(row["id"]))
            gt = [annotation_mask(row, gray.shape) for row in labels]
            candidates = record["instances"]
            matrix = pairwise_iou([encode(mask) for mask in gt], [r["segmentation"] for r in candidates])
            counts["candidates"] += len(candidates)
            for index, candidate in enumerate(candidates):
                match = best_match(matrix, index)
                if match is None:
                    counts["unmatched_or_ambiguous"] += 1
                    continue
                gt_index, iou = match
                target = gt[gt_index]
                prompt = decode(candidate["segmentation"]).astype(bool)
                try:
                    crop = plan_crop(prompt, context_px=32)
                except ValueError:
                    counts["unsupported_crop"] += 1
                    continue
                y = crop_array(target, crop)
                if not np.array_equal(paste_array(y, crop), target):
                    counts["target_outside_crop"] += 1
                    continue
                x = make_input(gray, probability, prompt, crop)
                key = f"{len(rows):06d}"
                relative = f"samples/{key}.npz"
                sample = stage / relative
                np.savez_compressed(sample, X=x, Y=y.astype(np.float32)[None])
                rows.append({"sample_id": key, "observation_id": obs, "candidate_id": candidate["instance_id"],
                             "annotation_id": labels[gt_index]["id"], "candidate_iou": iou, "kind": "matched",
                             "transform": crop.to_dict(), "npz": relative, "npz_sha256": sha256(sample)})
        if not rows:
            raise ValueError("No matched positive crops")
        manifest = stage / "manifest.jsonl"
        write_jsonl(manifest, rows)
        report = {"images": len(records), "positive_samples": len(rows), "counts": dict(counts),
                  "coarse_sha256": sha256(args.coarse), "annotation_sha256": sha256(args.annotations),
                  "protocol_sha256": sha256(PROTOCOL), "manifest_sha256": sha256(manifest), "is_oof": False}
        write_json(stage / "report.json", report)
        stage.rename(destination)
    print(json.dumps(report))


def observations_for(args):
    protocol = load_json(PROTOCOL)["observations"]
    if args.observations is None:
        return {key for key, value in protocol.items() if value["split"] == "holdout"}
    chosen = load_json(args.observations)
    if not isinstance(chosen, list) or not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(protocol):
        raise ValueError("Expected a JSON list of distinct observation IDs from the protocol")
    return set(chosen)


def evaluate(args):
    protocol = load_json(PROTOCOL)["observations"]
    expected = observations_for(args)
    predictions = {}
    sources = {}
    for item in args.prediction:
        name, sep, path = item.partition("=")
        if not sep or not name or name in predictions or name in ("observation_id", "annotation_record_id", "gt_instances"):
            raise ValueError("Each prediction must have a distinct NAME=JSONL")
        records = unique_records(path)
        if not expected <= set(records):
            raise ValueError(f"Prediction cohort incomplete for {name}")
        predictions[name] = {o: records[o] for o in sorted(expected)}
        sources[name] = sha256(path)
    reference_predictions = next(iter(predictions.values()))
    for name, records in predictions.items():
        if any(record["image_size"] != reference_predictions[o]["image_size"] for o, record in records.items()):
            raise ValueError(f"Original coordinates differ for {name}")
    images, labels = annotation_index(args.annotations, reference_predictions, protocol)
    passes = []
    for key, observation in sorted(images.items()):
        shape = reference_predictions[observation]["image_size"]
        gt = [encode(annotation_mask(row, shape)) for row in sorted(labels[key], key=lambda row: str(row["id"]))]
        if not gt:
            continue
        result = {"observation_id": observation, "annotation_record_id": key, "gt_instances": len(gt)}
        for name, records in predictions.items():
            instances = records[observation]["instances"]
            result[name] = pq_counts(pairwise_iou(gt, [row["segmentation"] for row in instances]))
        passes.append(result)
    if not passes:
        raise ValueError("No labelled annotation passes")
    report = {"observations": len(expected), "annotation_passes": len(passes),
              "gt_instances": sum(row["gt_instances"] for row in passes),
              "metrics": {a: aggregate_counts([r[a] for r in passes]) for a in predictions},
              "prediction_sha256": sources, "annotation_sha256": sha256(args.annotations)}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "per_annotation.jsonl", passes)
    write_json(out / "report.json", report)
    print(json.dumps(report))
