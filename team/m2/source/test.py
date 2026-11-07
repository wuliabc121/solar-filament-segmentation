#!/usr/bin/env python
"""M2 - inference, unified team evaluator and analysis for the SegFormer route.

Sub-commands
  selfcheck  evaluator boundary cases, RLE round trip, agreement with the official notebook code
  predict    sliding-window probabilities  -> semantic_probs_<subset>_<run_id>_<ckpt>.npz
  evaluate   PQ/SQ/RQ + error slices of a probability cache (threshold sweep only on calib)
  compare    E1 (M1 CNN) vs E2 (SegFormer) and the E4 fusion sweep P = a*P_C + (1-a)*P_T
  score      team scoring entry: any submission-format CSV vs ground truth (official logic)
  export     probability cache + frozen post-processing -> submission-format CSV

Examples (see README.md):
  python test.py selfcheck
  python test.py predict  --ckpt runs/<run>/best.pt --subset calib
  python test.py evaluate --probs runs/<run>/preds/semantic_probs_calib_<...>.npz --sweep
  python test.py evaluate --probs runs/<run>/preds/semantic_probs_holdout_<...>.npz --postprocess <pp.json>
  python test.py compare  --cnn <M1 calib npz> --transformer <M2 calib npz> --postprocess <pp.json>
  python test.py score    --pred submission.csv --split <split_manifest.json> --subset calib
  python test.py export   --probs <test npz> --postprocess <pp.json>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_util

from train import (DEFAULT_ANN, H, LARGE_AREA, OFFICIAL_EVAL, ROOT, SMALL_AREA, TEST_IMG_DIR, TRAIN_IMG_DIR, W,
                   NpzWriter, add_counts, build_model, components, env_info, gt_for_stem, iou_labelmap,
                   load_annotations, load_split, official_counts, polygon_rle, pq_summary, predict_prob,
                   read_gray, score_detail, score_grid, semantic_label, sha256_file, subset_stems)

ZERO_BELOW = 1e-3  # cached probabilities below this are stored as exact 0 (keeps the npz small)
SLICES = ("small", "medium", "large", "slender", "adjacent")


# --------------------------------------------------------------------------- helpers
def open_probs(path):
    """Probability cache -> (meta, stems, getter). Keys are observation stems, values HxW probabilities."""
    z = np.load(path)
    meta = json.loads(str(z["__meta__"]))
    stems = sorted(k for k in z.files if k != "__meta__")
    return meta, stems, lambda s: z[s].astype(np.float32)


def in_slice(g, name):
    return (not g["small"] and not g["large"]) if name == "medium" else g[name]


def mask_to_counts(mask):
    """Binary HxW mask -> COCO compressed RLE counts string (column-major, as pycocotools expects)."""
    counts = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")
    if set(counts) & set(',"\'\r\n'):
        raise ValueError("RLE counts contain CSV-hostile characters")
    return counts


def component_counts(prob, thr, min_area):
    """Fixed basic instantiation -> list of RLE counts strings (one per kept component)."""
    lab, areas = components(prob, thr)
    lab = lab.reshape(prob.shape)
    return [mask_to_counts(lab == j + 1) for j in np.flatnonzero(areas >= min_area)]


def read_rows(path):
    with open(path, newline="") as f:
        return [(r["filament_id"], r["segmentation_rle"]) for r in csv.DictReader(f)]


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filament_id", "segmentation_rle"])
        w.writerows(rows)


def gt_rows_from_json(obs, stems):
    """Official GT format: filament_id = '<record_id>_<k>' (record_id = '<annotator>-<image stem>')."""
    rows = []
    for s in stems:
        for rid, anns in sorted(obs[s]["records"].items()):
            for k, a in enumerate(anns, 1):
                rows.append((f"{rid}_{k}", polygon_rle(a)["counts"].decode("ascii")))
    return rows


def score_csv(gt_rows, pred_rows, h=H, w=W, per_record=False):
    """Official notebook logic on (filament_id, rle counts) rows; IoU via pycocotools on RLEs.

    GT is grouped by annotator-image (filament_id before the first '_'); each group is matched
    against all predictions whose image id (prediction filament_id before '_') equals the part of
    the annotator-image after the first '-'. Predictions of images without GT rows are ignored.
    """
    gt_by, pred_by = defaultdict(list), defaultdict(list)
    for fid, c in gt_rows:
        gt_by[fid.split("_", 1)[0]].append({"size": [h, w], "counts": c.encode("ascii")})
    for fid, c in pred_rows:
        pred_by[fid.split("_", 1)[0]].append({"size": [h, w], "counts": c.encode("ascii")})
    acc, records = {}, {}
    for ai, g in gt_by.items():
        p = pred_by.get(ai.split("-", 1)[1], [])
        iou = (np.nan_to_num(np.asarray(mask_util.iou(p, g, [0] * len(g)), np.float64).T)
               if g and p else np.zeros((len(g), len(p))))
        c = official_counts(iou)
        add_counts(acc, c)
        records[ai] = c
    out = pq_summary(acc)
    if per_record:
        out["records"] = records
    return out


def bootstrap_diff(counts_a, counts_b, n_boot=2000, seed=0):
    """Paired bootstrap over observations of pooled PQ(b) - PQ(a); counts arrays are (n_obs, 4)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(counts_a), size=(n_boot, len(counts_a)))

    def pq(c):
        s = c[idx].sum(1)
        den = s[:, 0] + 0.5 * s[:, 1] + 0.5 * s[:, 2]
        return np.where(den > 0, s[:, 3] / np.maximum(den, 1e-12), 0.0)

    d = pq(counts_b) - pq(counts_a)
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "n_boot": n_boot, "seed": seed}


def as_array(per_obs, stems):
    return np.array([[per_obs[s][k] for k in ("tp", "fp", "fn", "iou_sum")] for s in stems], np.float64)


# --------------------------------------------------------------------------- predict
def cmd_predict(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    run_dir = Path(a.ckpt).resolve().parent
    if a.subset == "test":
        files = sorted(TEST_IMG_DIR.glob("*.jpeg"))
        stems, split_hash = [f.stem for f in files], None
    else:
        split = load_split(a.split or run_dir / "split_manifest.json")
        if split["split_hash"] != ck["split_hash"]:
            raise ValueError("split_hash of --split differs from the checkpoint's training split")
        if a.subset == "oof" and ck["fold"] != a.fold:
            raise ValueError(f"OOF fold {a.fold} needs a checkpoint trained with --fold {a.fold} "
                             f"(this one: fold={ck['fold']}); otherwise the model has seen these observations")
        obs = load_annotations(a.ann)
        stems, split_hash = subset_stems(split, a.subset, a.fold), split["split_hash"]
        files = [TRAIN_IMG_DIR / obs[s]["file"] for s in stems]
    if a.limit:
        stems, files = stems[:a.limit], files[:a.limit]

    model = build_model(hf_config=ck["hf_config"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    out_dir = Path(a.out_dir) if a.out_dir else run_dir / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.subset + ("" if a.fold is None or a.subset != "oof" else f"_fold{a.fold}")
    out = out_dir / f"semantic_probs_{tag}_{ck['run_id']}_{Path(a.ckpt).stem}.npz"
    if out.exists():
        raise FileExistsError(f"{out} exists; caches are never overwritten (new checkpoint/config -> new file)")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    writer, t0 = NpzWriter(out), time.time()
    try:
        for i, (s, f) in enumerate(zip(stems, files), 1):
            p = predict_prob(model, read_gray(f), device, a.tile, a.overlap, a.batch)
            p[p < ZERO_BELOW] = 0
            writer.add(s, p.astype(np.float16))
            if i % 20 == 0 or i == len(stems):
                print(f"  {i}/{len(stems)}  {(time.time() - t0) / i:.2f}s/img", flush=True)
        meta = {"producer": "M2", "model": "segformer", "run_id": ck["run_id"],
                "checkpoint": str(Path(a.ckpt).resolve()), "checkpoint_sha256": sha256_file(a.ckpt),
                "checkpoint_epoch": ck["epoch"], "subset": a.subset, "fold": a.fold if a.subset == "oof" else None,
                "split_hash": split_hash, "n_images": len(stems), "dtype": "float16", "zero_below": ZERO_BELOW,
                "coords": "original image pixels, origin top-left, array HxW = (row, col)", "size": [H, W],
                "transform": {"resize": None, "tile": a.tile, "overlap": a.overlap,
                              "blend": "Hann-weighted mean of tile probabilities", "tta": None},
                "sec_per_image": round((time.time() - t0) / max(1, len(stems)), 3),
                "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
                "env": env_info()}
        writer.add("__meta__", np.array(json.dumps(meta)))
        writer.close()
    except BaseException:
        writer.close()
        out.unlink(missing_ok=True)
        raise
    print(f"saved {out}")


# --------------------------------------------------------------------------- evaluate
def detail_metrics(obs, stems, get_prob, thr, min_area):
    total, per_obs, hits, fp_areas = {}, {}, [], []
    extra = defaultdict(int)
    for s in stems:
        d = score_detail(get_prob(s), gt_for_stem(obs[s]), thr, min_area)
        add_counts(total, d["counts"])
        per_obs[s] = pq_summary(d["counts"])
        hits += d["gt_hits"]
        fp_areas += d["fp_areas"]
        for k in ("gt_split", "pred_merge", "dice_inter", "dice_pred", "dice_gt", "n_pred"):
            extra[k] += d[k]
    slices = {"all": {"n_gt": len(hits), "recall": float(np.mean([h["hit"] for h in hits])) if hits else 0.0}}
    for name in SLICES:
        sel = [h["hit"] for h in hits if in_slice(h, name)]
        slices[name] = {"n_gt": len(sel), "recall": float(np.mean(sel)) if sel else 0.0}
    fp = np.asarray(fp_areas)
    inter, pred, gt = extra["dice_inter"], extra["dice_pred"], extra["dice_gt"]
    return {
        **pq_summary(total),
        "dice": 2 * inter / (pred + gt) if pred + gt else 0.0,
        "iou": inter / (pred + gt - inter) if pred + gt - inter else 0.0,
        "n_observations": len(stems), "n_records": len({h["record"] for h in hits}), "n_pred_instances": extra["n_pred"],
        "gt_slices": slices,
        "fp_by_area": {"small": int((fp < SMALL_AREA).sum()), "medium": int(((fp >= SMALL_AREA) & (fp < LARGE_AREA)).sum()),
                       "large": int((fp >= LARGE_AREA).sum())},
        "gt_split_1_to_n": extra["gt_split"], "pred_merge_n_to_1": extra["pred_merge"],
        "per_observation": per_obs,
    }, hits


def cmd_evaluate(a):
    meta, stems, get = open_probs(a.probs)
    if meta["subset"] == "test":
        raise ValueError("test images have no ground truth; use 'export'")
    obs = load_annotations(a.ann)
    t0 = time.time()
    grid_table = None
    if a.sweep:
        if meta["subset"] == "holdout":
            raise ValueError("the holdout set must not be used to select thresholds (sweep on calib only)")
        thrs = [float(v) for v in a.thrs.split(",")]
        mas = [int(v) for v in a.min_areas.split(",")]
        acc = defaultdict(dict)
        for i, s in enumerate(stems, 1):
            for key, c in score_grid(get(s), gt_for_stem(obs[s]), thrs, mas).items():
                add_counts(acc[key], c)
            if i % 20 == 0:
                print(f"  sweep {i}/{len(stems)}", flush=True)
        grid_table = [{"thr": t, "min_area": m, **pq_summary(acc[(t, m)])} for t in thrs for m in mas]
        best = max(grid_table, key=lambda r: r["PQ"])
        pp = {"thr": best["thr"], "min_area": best["min_area"], "connectivity": 8, "rule": "prob >= thr",
              "selected_on": {"probs": Path(a.probs).name, "subset": meta["subset"], "run_id": meta["run_id"],
                              "split_hash": meta["split_hash"], "criterion": "max pooled PQ"},
              "grid": {"thrs": thrs, "min_areas": mas}}
        pp_path = Path(a.probs).with_name(f"postprocess_{meta['run_id']}.json")
        pp_path.write_text(json.dumps(pp, indent=1))
        print(f"selected thr={pp['thr']} min_area={pp['min_area']} (calib PQ {best['PQ']:.4f}) -> {pp_path}")
    elif a.postprocess:
        pp = json.loads(Path(a.postprocess).read_text())
    else:
        pp = {"thr": a.thr, "min_area": a.min_area, "connectivity": 8, "rule": "prob >= thr", "selected_on": None}

    m, _ = detail_metrics(obs, stems, get, pp["thr"], pp["min_area"])
    metrics = {"probs": str(Path(a.probs).resolve()), "probs_meta": {k: meta.get(k) for k in (
                   "run_id", "checkpoint_sha256", "checkpoint_epoch", "subset", "fold", "split_hash")},
               "postprocess": pp, "evaluator": OFFICIAL_EVAL, "annotation_sha256": sha256_file(a.ann),
               "eval_sec": round(time.time() - t0, 1), "infer_sec_per_image": meta.get("sec_per_image"),
               "peak_gpu_mem_gb": meta.get("peak_gpu_mem_gb"), **m, "sweep": grid_table}
    out = Path(a.out) if a.out else Path(a.probs).with_name(f"metrics_{Path(a.probs).stem.removeprefix('semantic_probs_')}.json")
    out.write_text(json.dumps(metrics, indent=1))
    print(f"[{meta['subset']}] PQ {m['PQ']:.4f}  SQ {m['SQ']:.4f}  RQ {m['RQ']:.4f}  TP {m['tp']} FP {m['fp']} FN {m['fn']}  "
          f"Dice {m['dice']:.4f}  IoU {m['iou']:.4f}  (thr={pp['thr']}, min_area={pp['min_area']})")
    print("  recall by slice: " + "  ".join(f"{k} {v['recall']:.3f}/{v['n_gt']}" for k, v in m["gt_slices"].items()))
    print(f"  FP by area {m['fp_by_area']}  1:n splits {m['gt_split_1_to_n']}  n:1 merges {m['pred_merge_n_to_1']}")
    print(f"saved {out}")


# --------------------------------------------------------------------------- compare (E1 / E2 / E4)
def cmd_compare(a):
    mc, sc, get_c = open_probs(a.cnn)
    mt, st, get_t = open_probs(a.transformer)
    if sc != st or mc["subset"] != mt["subset"] or mc["split_hash"] != mt["split_hash"]:
        raise ValueError("CNN and Transformer caches must cover the same observations of the same split/subset")
    if mc["subset"] in ("test", "holdout") and not a.allow_holdout:
        raise ValueError("compare/fusion selection runs on calib; use --allow-holdout only for the frozen final check")
    pp = json.loads(Path(a.postprocess).read_text()) if a.postprocess else {"thr": a.thr, "min_area": a.min_area}
    alphas = [float(v) for v in a.alphas.split(",")]
    obs = load_annotations(a.ann)
    total = {al: {} for al in alphas}
    per_obs = {al: {} for al in alphas}
    hits = {}
    for i, s in enumerate(sc, 1):
        gt = gt_for_stem(obs[s])
        pc, pt = get_c(s), get_t(s)
        for al in alphas:
            d = score_detail(al * pc + (1 - al) * pt, gt, pp["thr"], pp["min_area"])
            add_counts(total[al], d["counts"])
            per_obs[al][s] = {k: d["counts"].get(k, 0) for k in ("tp", "fp", "fn", "iou_sum")}
            if al in (0.0, 1.0):
                hits.setdefault(al, []).extend(d["gt_hits"])
        if i % 20 == 0:
            print(f"  {i}/{len(sc)}", flush=True)
    table = [{"alpha": al, **pq_summary(total[al])} for al in alphas]
    result = {"cnn": {k: mc.get(k) for k in ("run_id", "checkpoint_sha256")},
              "transformer": {k: mt.get(k) for k in ("run_id", "checkpoint_sha256")},
              "subset": mc["subset"], "split_hash": mc["split_hash"], "postprocess": pp, "evaluator": OFFICIAL_EVAL,
              "fusion": "P = alpha * P_CNN + (1 - alpha) * P_Transformer (alpha=1: E1, alpha=0: E2)",
              "alpha_table": table}
    if 0.0 in hits and 1.0 in hits:
        hc, ht = hits[1.0], hits[0.0]

        def comp(sel):
            pairs = [(c["hit"], t["hit"]) for c, t in zip(hc, ht) if sel(c)]
            return {"n_gt": len(pairs), "both": sum(c and t for c, t in pairs),
                    "only_cnn": sum(c and not t for c, t in pairs),
                    "only_transformer": sum(t and not c for c, t in pairs),
                    "neither": sum(not c and not t for c, t in pairs)}
        result["complementarity"] = {"all": comp(lambda g: True),
                                     **{n: comp(lambda g, n=n: in_slice(g, n)) for n in SLICES}}
        ca, ct = as_array(per_obs[1.0], sc), as_array(per_obs[0.0], sc)
        pq_obs = lambda c: np.where(c[:, 0] + 0.5 * c[:, 1] + 0.5 * c[:, 2] > 0,
                                    c[:, 3] / np.maximum(c[:, 0] + 0.5 * c[:, 1] + 0.5 * c[:, 2], 1e-12), 0.0)
        diff = pq_obs(ct) - pq_obs(ca)
        result["per_observation_transformer_minus_cnn"] = {
            "better": int((diff > 1e-9).sum()), "worse": int((diff < -1e-9).sum()), "equal": int((np.abs(diff) <= 1e-9).sum()),
            "values": dict(zip(sc, np.round(diff, 5).tolist()))}
        result["bootstrap_pooled_PQ_transformer_minus_cnn"] = bootstrap_diff(ca, ct)
    out = Path(a.out) if a.out else Path(a.transformer).with_name(f"compare_{mc['subset']}_{mt['run_id']}.json")
    out.write_text(json.dumps(result, indent=1))
    for r in table:
        print(f"alpha {r['alpha']:.2f}: PQ {r['PQ']:.4f}  SQ {r['SQ']:.4f}  RQ {r['RQ']:.4f}  TP {r['tp']} FP {r['fp']} FN {r['fn']}")
    if "complementarity" in result:
        print("complementarity (GT hits):", result["complementarity"]["all"])
        print("bootstrap PQ(T) - PQ(C):", result["bootstrap_pooled_PQ_transformer_minus_cnn"])
    print(f"saved {out}")


# --------------------------------------------------------------------------- score / export
def cmd_score(a):
    pred = read_rows(a.pred)
    if a.gt:
        gt = read_rows(a.gt)
    else:
        if not (a.split and a.subset):
            raise ValueError("give --gt CSV, or --split and --subset to build GT from the annotation json")
        obs = load_annotations(a.ann)
        gt = gt_rows_from_json(obs, subset_stems(load_split(a.split), a.subset, a.fold))
        if a.save_gt:
            write_rows(a.save_gt, gt)
            print(f"saved GT CSV {a.save_gt}")
    r = score_csv(gt, pred)
    print(f"PQ {r['PQ']:.4f}  SQ {r['SQ']:.4f}  RQ {r['RQ']:.4f}  TP {r['tp']} FP {r['fp']} FN {r['fn']}  "
          f"({len(gt)} GT rows, {len(pred)} prediction rows)")
    if a.out:
        Path(a.out).write_text(json.dumps({"pred": str(a.pred), "evaluator": OFFICIAL_EVAL, **r}, indent=1))


def cmd_export(a):
    meta, stems, get = open_probs(a.probs)
    pp = json.loads(Path(a.postprocess).read_text())
    rows = []
    for s in stems:
        rows += [(f"{s}_{k}", c) for k, c in enumerate(component_counts(get(s), pp["thr"], pp["min_area"]), 1)]
    out = Path(a.out) if a.out else Path(a.probs).with_name(f"submission_{Path(a.probs).stem.removeprefix('semantic_probs_')}.csv")
    write_rows(out, rows)
    print(f"{len(rows)} instances from {len(stems)} images (thr={pp['thr']}, min_area={pp['min_area']}) -> {out}")


# --------------------------------------------------------------------------- selfcheck
def official_dense_iou(gt_masks, pred_masks):
    """Numpy port of get_overlap_matrices() from the official notebook (dense masks, matmul)."""
    g = np.stack(gt_masks).reshape(len(gt_masks), -1).astype(np.float32)
    p = (np.stack(pred_masks).reshape(len(pred_masks), -1).astype(np.float32) if pred_masks
         else np.zeros((0, g.shape[1]), np.float32))
    inter = g @ p.T
    union = g.sum(1)[:, None] + p.sum(1)[None] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union == 0, 0.0, inter / union)


def cmd_selfcheck(a):
    report, ok = [], True

    def check(name, cond, info=None):
        nonlocal ok
        ok &= bool(cond)
        report.append({"check": name, "pass": bool(cond), "info": info})
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {info}" if info is not None else ""))

    # 1) RLE round trip with an asymmetric mask (catches transposed encodings)
    m = np.zeros((H, W), np.uint8)
    m[10:20, 100:400] = 1
    m[500:900, 30:35] = 1
    back = mask_util.decode({"size": [H, W], "counts": mask_to_counts(m).encode()})
    check("RLE round trip (asymmetric mask)", np.array_equal(back, m))
    wrong = mask_util.decode(mask_util.encode(np.asfortranarray(m.T)))
    check("transposed mask is detected as different", not np.array_equal(wrong, m))

    # 2) hand-derived boundary cases on a small 8x10 canvas
    h, w = 8, 10

    def box(y0, x0, y1, x1):
        b = np.zeros((h, w), np.uint8)
        b[y0:y1, x0:x1] = 1
        return b
    A, A_left, A_right, A6 = box(0, 0, 2, 4), box(0, 0, 2, 2), box(0, 2, 2, 4), box(0, 0, 2, 3)
    far = box(5, 6, 7, 9)
    cases = [  # name, {record: [gt masks]}, {image: [pred masks]}, expected (tp, fp, fn, iou_sum)
        ("perfect match", {"r1-img": [A]}, {"img": [A]}, (1, 0, 0, 1.0)),
        ("empty prediction -> all FN", {"r1-img": [A, far]}, {}, (0, 0, 2, 0.0)),
        ("IoU exactly 0.5 is not a match", {"r1-img": [A]}, {"img": [A_left]}, (0, 1, 1, 0.0)),
        ("IoU 0.75 is a match", {"r1-img": [A]}, {"img": [A6]}, (1, 0, 0, 0.75)),
        ("one GT split in two halves", {"r1-img": [A]}, {"img": [A_left, A_right]}, (0, 2, 1, 0.0)),
        ("two GT merged into one prediction", {"r1-img": [A_left, A_right]}, {"img": [A]}, (0, 1, 2, 0.0)),
        ("two annotators pooled", {"r1-img": [A], "r2-img": [A6]}, {"img": [A]}, (2, 0, 0, 1.75)),
        ("extra prediction is FP", {"r1-img": [A]}, {"img": [A, far]}, (1, 1, 0, 1.0)),
        ("predictions of images without GT are ignored", {"r1-img": [A]}, {"other": [A]}, (0, 0, 1, 0.0)),
    ]
    for name, gts, preds, exp in cases:
        gt_rows = [(f"{r}_{k}", mask_to_counts(g)) for r, gs in gts.items() for k, g in enumerate(gs, 1)]
        pr_rows = [(f"{i}_{k}", mask_to_counts(p)) for i, ps in preds.items() for k, p in enumerate(ps, 1)]
        r_csv = score_csv(gt_rows, pr_rows, h, w)
        acc_lab, acc_dense = {}, {}
        for r, gs in gts.items():
            ps = preds.get(r.split("-", 1)[1], [])
            lab = np.zeros(h * w, np.int64)
            for j, p in enumerate(ps, 1):
                lab[p.ravel() > 0] = j
            areas = np.array([p.sum() for p in ps], np.int64)
            insts = [{"idx": np.flatnonzero(g), "area": int(g.sum())} for g in gs]
            add_counts(acc_lab, official_counts(iou_labelmap(insts, lab, areas)))
            add_counts(acc_dense, official_counts(official_dense_iou(gs, ps)))
        got = [tuple(round(c.get(k, 0), 6) for k in ("tp", "fp", "fn", "iou_sum")) for c in (r_csv, acc_lab, acc_dense)]
        check(f"case: {name}", all(g == exp for g in got), {"expected": exp, "csv/labelmap/dense": got})

    # 3) real annotations: three IoU paths agree, and export -> score equals evaluate
    obs = load_annotations(a.ann)
    stems = [s for s in sorted(obs) if len(obs[s]["records"]) >= 2][:a.n_real]
    for s in stems:
        prob = semantic_label(obs[s])  # stand-in "prediction": mean annotator label
        gt = gt_for_stem(obs[s])
        lab, areas = components(prob, 0.5)
        pred_masks = [(lab == j + 1).reshape(H, W) for j in range(len(areas))]
        same = True
        for rid, insts in gt.items():
            gt_masks = [mask_util.decode(polygon_rle(x)) for x in obs[s]["records"][rid]]
            m_lab = iou_labelmap(insts, lab, areas)
            m_dense = official_dense_iou(gt_masks, pred_masks)
            g_r = [polygon_rle(x) for x in obs[s]["records"][rid]]
            p_r = [mask_util.encode(np.asfortranarray(p.astype(np.uint8))) for p in pred_masks]
            m_rle = np.asarray(mask_util.iou(p_r, g_r, [0] * len(g_r))).T
            same &= np.allclose(m_lab, m_dense, atol=1e-6) and np.allclose(m_lab, m_rle, atol=1e-6)
            same &= all(official_counts(m_lab)[k] == official_counts(m_dense)[k] for k in ("tp", "fp", "fn"))
        check(f"real {s}: labelmap / official-dense / RLE IoU agree ({len(gt)} records)", same)
        r_eval = pq_summary(score_grid(prob, gt, [0.5], [0])[(0.5, 0)])
        r_csv = score_csv(gt_rows_from_json(obs, [s]), [(f"{s}_{k}", c) for k, c in
                                                         enumerate(component_counts(prob, 0.5, 0), 1)])
        check(f"real {s}: export->score PQ == evaluate PQ", abs(r_eval["PQ"] - r_csv["PQ"]) < 1e-9,
              {"evaluate": round(r_eval["PQ"], 6), "score_csv": round(r_csv["PQ"], 6)})

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"all_pass": ok, "evaluator": OFFICIAL_EVAL, "checks": report}, indent=1, default=str))
    print(f"{'ALL PASS' if ok else 'SOME CHECKS FAILED'} -> {out}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- cli
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selfcheck", help="evaluator boundary tests")
    s.add_argument("--ann", default=str(DEFAULT_ANN))
    s.add_argument("--n-real", type=int, default=3, help="real multi-annotator observations to cross-check")
    s.add_argument("--out", default=str(ROOT / "temp/selfcheck.json"))

    s = sub.add_parser("predict", help="sliding-window probabilities -> npz cache")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--subset", required=True, choices=["calib", "holdout", "oof", "test"])
    s.add_argument("--fold", type=int, default=None, help="inner fold for --subset oof")
    s.add_argument("--split", default=None, help="default: split_manifest.json next to the checkpoint")
    s.add_argument("--ann", default=str(DEFAULT_ANN))
    s.add_argument("--tile", type=int, default=512)
    s.add_argument("--overlap", type=int, default=128)
    s.add_argument("--batch", type=int, default=4)
    s.add_argument("--limit", type=int, default=None, help="first N images only (smoke test)")
    s.add_argument("--out-dir", default=None, help="default: <run dir>/preds")

    s = sub.add_parser("evaluate", help="metrics for a probability cache")
    s.add_argument("--probs", required=True)
    s.add_argument("--ann", default=str(DEFAULT_ANN))
    s.add_argument("--sweep", action="store_true", help="select thr/min_area on this (calib) cache")
    s.add_argument("--thrs", default="0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7")
    s.add_argument("--min-areas", default="0,50,100,200,400,800")
    s.add_argument("--postprocess", default=None, help="frozen postprocess json (from a calib sweep)")
    s.add_argument("--thr", type=float, default=0.5)
    s.add_argument("--min-area", type=int, default=200)
    s.add_argument("--out", default=None)

    s = sub.add_parser("compare", help="E1 vs E2 and E4 fusion alpha sweep")
    s.add_argument("--cnn", required=True, help="M1 probability cache (same npz format)")
    s.add_argument("--transformer", required=True, help="M2 probability cache")
    s.add_argument("--postprocess", default=None, help="the shared fixed thr/min_area for E1/E2/E4")
    s.add_argument("--thr", type=float, default=0.5)
    s.add_argument("--min-area", type=int, default=200)
    s.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    s.add_argument("--allow-holdout", action="store_true")
    s.add_argument("--ann", default=str(DEFAULT_ANN))
    s.add_argument("--out", default=None)

    s = sub.add_parser("score", help="score a submission-format CSV (official logic)")
    s.add_argument("--pred", required=True)
    s.add_argument("--gt", default=None, help="GT CSV in official format; or use --split/--subset")
    s.add_argument("--split", default=None)
    s.add_argument("--subset", default=None, choices=["calib", "holdout", "oof", "train"])
    s.add_argument("--fold", type=int, default=None)
    s.add_argument("--save-gt", default=None, help="also write the GT CSV (for the official notebook)")
    s.add_argument("--ann", default=str(DEFAULT_ANN))
    s.add_argument("--out", default=None)

    s = sub.add_parser("export", help="probabilities + frozen postprocess -> submission-format CSV")
    s.add_argument("--probs", required=True)
    s.add_argument("--postprocess", required=True)
    s.add_argument("--out", default=None)

    a = p.parse_args(argv)
    return {"selfcheck": cmd_selfcheck, "predict": cmd_predict, "evaluate": cmd_evaluate,
            "compare": cmd_compare, "score": cmd_score, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
