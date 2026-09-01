"""Verify-first EfficientNet forensic audit. Does not train or overwrite weights."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.models.canonical_paths import (
    EFFNET_CANDIDATE,
    EFFNET_CANONICAL,
    EFFNET_METADATA,
    REGISTRY_PATH,
    abs_path,
    posix,
    read_json,
    sha256_file,
)
from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier
from ml.vision.input_quality import assess_input_quality

ROOT_DS = os.path.join("data", "datasets", "public_wound_dataset")
MANIFEST = os.path.join(ROOT_DS, "manifest.csv")
OUT_JSON = os.path.join("scratch", "forensic_efficientnet_audit.json")
OUT_TXT = os.path.join("scratch", "forensic_efficientnet_audit.txt")


def _pixel_sha(bgr: np.ndarray) -> str:
    return hashlib.sha256(bgr.tobytes() + f"|{bgr.shape[0]}x{bgr.shape[1]}".encode()).hexdigest()


def _dhash(bgr: np.ndarray, size: int = 8) -> str:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    return "".join("1" if v else "0" for v in diff.flatten())


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def _round_probs(d: dict) -> dict:
    return {k: (None if v is None else round(float(v), 6)) for k, v in d.items()}


def artifact_identity():
    path = abs_path(EFFNET_CANONICAL)
    disk_sha = sha256_file(path)
    meta = read_json(EFFNET_METADATA)
    registry = read_json(REGISTRY_PATH)
    reg = (registry or {}).get("EfficientNetV2 Classification") or {}
    state = torch.load(path, map_location="cpu")
    n_classes = None
    head_key = None
    for key, tensor in state.items():
        if str(key).endswith("classifier.weight"):
            n_classes = int(tensor.shape[0])
            head_key = key
            break
    cand_sha = sha256_file(EFFNET_CANDIDATE) if os.path.isfile(abs_path(EFFNET_CANDIDATE)) else None
    backup = abs_path(EFFNET_CANONICAL + ".pre_retrain_backup")
    return {
        "canonical_path": posix(EFFNET_CANONICAL),
        "exists": os.path.isfile(path),
        "file_size": os.path.getsize(path),
        "sha256": disk_sha,
        "metadata_sha256": meta.get("artifact_sha256"),
        "registry_sha256": reg.get("artifact_sha256"),
        "sha_matches_metadata": disk_sha == meta.get("artifact_sha256"),
        "sha_matches_registry": disk_sha == reg.get("artifact_sha256"),
        "metadata_status": meta.get("status"),
        "metadata_classes": meta.get("classes"),
        "n_classes_from_checkpoint": n_classes,
        "classifier_weight_key": head_key,
        "candidate_path": posix(EFFNET_CANDIDATE),
        "candidate_sha256": cand_sha,
        "candidate_equals_production": cand_sha == disk_sha if cand_sha else False,
        "backup_exists": os.path.isfile(backup),
        "backup_sha256": sha256_file(backup) if os.path.isfile(backup) else None,
        "did_not_retrain": True,
    }


def audit_dataset():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    by_split = defaultdict(list)
    by_class = Counter()
    split_class = defaultdict(Counter)
    subjects = defaultdict(set)
    hashes = defaultdict(list)
    dhashes = defaultdict(list)
    missing = 0
    unique_by_class = defaultdict(set)
    raw_label_remap_note = (
        "download_public_datasets.py draws abrasion/laceration/burn then writes class='swelling'. "
        "Manifest source claims Kaggle/Roboflow/WOUNDSEG. Images are PIL drawings."
    )
    for row in rows:
        split = row["split"]
        cls = row["class"]
        by_split[split].append(row)
        by_class[cls] += 1
        split_class[split][cls] += 1
        subjects[split].add(row.get("subject_id") or "")
        path = os.path.join(ROOT_DS, row["image_path"].replace("/", os.sep))
        if not os.path.exists(path):
            missing += 1
            continue
        bgr = cv2.imread(path)
        if bgr is None:
            missing += 1
            continue
        digest = _pixel_sha(bgr)
        hashes[digest].append((split, row["sample_id"], cls, row.get("subject_id")))
        dhashes[_dhash(bgr)].append((split, row["sample_id"], cls))
        unique_by_class[cls].add(digest)

    exact_dup_groups = {h: v for h, v in hashes.items() if len(v) > 1}
    leak_exact = []
    for digest, group in exact_dup_groups.items():
        splits = {s for s, _, _, _ in group}
        if len(splits) > 1:
            leak_exact.append({
                "pixel_sha256": digest,
                "n": len(group),
                "splits": sorted(splits),
                "classes": sorted({c for _, _, c, _ in group}),
                "subjects": sorted({sid for _, _, _, sid in group}),
                "sample_ids_preview": [sid for _, sid, _, _ in group[:8]],
            })

    leak_near = []
    keys = list(dhashes.keys())
    # exact dHash groups first
    near_groups = {h: v for h, v in dhashes.items() if len(v) > 1}
    for h, group in near_groups.items():
        splits = {s for s, _, _ in group}
        if len(splits) > 1:
            leak_near.append({
                "dhash": h,
                "count": len(group),
                "splits": sorted(splits),
                "classes": sorted({c for _, _, c in group}),
            })

    # pairwise hamming <= 6 across unique dhashes
    close_pairs = 0
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if _hamming(a, b) <= 6:
                close_pairs += 1

    train_subj, val_subj, test_subj = subjects["train"], subjects["val"], subjects["test"]
    unique_templates = {k: len(v) for k, v in unique_by_class.items()}
    return {
        "root": posix(ROOT_DS),
        "provenance": "SYNTHETIC_PIL_DRAWINGS",
        "manifest_source_field_is_false_download_claim": True,
        "label_quality": raw_label_remap_note,
        "n_manifest": len(rows),
        "missing_images": missing,
        "class_balance_all": dict(by_class),
        "split_sizes": {k: len(v) for k, v in by_split.items()},
        "split_class_counts": {k: dict(v) for k, v in split_class.items()},
        "unique_pixel_templates_by_class": unique_templates,
        "unique_pixel_templates_total": sum(unique_templates.values()),
        "subject_counts": {k: len(v) for k, v in subjects.items()},
        "subject_overlap": {
            "train_val": sorted(train_subj & val_subj),
            "train_test": sorted(train_subj & test_subj),
            "val_test": sorted(val_subj & test_subj),
        },
        "subject_ids_are_synthetic_generator_labels": True,
        "exact_pixel_duplicate_groups": len(exact_dup_groups),
        "exact_duplicate_images": int(sum(len(v) for v in exact_dup_groups.values())),
        "cross_split_exact_duplicate_groups": leak_exact,
        "cross_split_exact_duplicate_group_count": len(leak_exact),
        "near_duplicate_exact_dhash_groups": len(near_groups),
        "cross_split_near_duplicate_dhash_groups": leak_near,
        "dhash_pairs_hamming_le_6": close_pairs,
        "negative_or_no_injury_labeled": int(by_class.get("none", 0) + by_class.get("normal", 0) + by_class.get("other", 0)),
        "needs_negative_class": True,
        "held_out_accuracy_is_not_generalization": True,
    }


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        acc = float((pred[mask] == labels[mask]).mean())
        mean_conf = float(conf[mask].mean())
        w = float(mask.sum()) / n
        ece += abs(acc - mean_conf) * w
        rows.append({
            "bin": [round(lo, 2), round(hi, 2)],
            "n": int(mask.sum()),
            "accuracy": round(acc, 6),
            "mean_confidence": round(mean_conf, 6),
        })
    return round(float(ece), 6), rows


def live_split_eval(clf: EfficientNetV2Classifier, split: str):
    rows = [r for r in csv.DictReader(open(MANIFEST, encoding="utf-8")) if r["split"] == split]
    classes = list(clf.classes)
    idx = {c: i for i, c in enumerate(classes)}
    ys, preds, maxps = [], [], []
    prob_mat = []
    unique_seen = set()
    for row in rows:
        path = os.path.join(ROOT_DS, row["image_path"].replace("/", os.sep))
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        unique_seen.add(_pixel_sha(bgr))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raw = clf.predict_raw(rgb, temperature=1.0)
        vec = np.array([raw["probs"][c.capitalize()] for c in classes], dtype=np.float64)
        y = idx[row["class"]]
        ys.append(y)
        preds.append(int(vec.argmax()))
        maxps.append(float(vec.max()))
        prob_mat.append(vec)
    ys_a = np.array(ys)
    preds_a = np.array(preds)
    probs = np.vstack(prob_mat) if prob_mat else np.zeros((0, len(classes)))
    acc = float(accuracy_score(ys_a, preds_a)) if len(ys_a) else None
    prec, rec, f1, support = precision_recall_fscore_support(
        ys_a, preds_a, labels=list(range(len(classes))), zero_division=0
    )
    cm = confusion_matrix(ys_a, preds_a, labels=list(range(len(classes)))).tolist()
    onehot = np.eye(len(classes))[ys_a] if len(ys_a) else np.zeros((0, len(classes)))
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1))) if len(ys_a) else None
    ece, bins = expected_calibration_error(probs, ys_a) if len(ys_a) else (None, [])
    per_class = {}
    for i, name in enumerate(classes):
        per_class[name] = {
            "precision": round(float(prec[i]), 6),
            "recall": round(float(rec[i]), 6),
            "f1": round(float(f1[i]), 6),
            "support": int(support[i]),
        }
    return {
        "split": split,
        "n": int(len(ys_a)),
        "unique_pixel_templates": len(unique_seen),
        "accuracy": None if acc is None else round(acc, 6),
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_labels": classes,
        "mean_max_softmax": None if not maxps else round(float(np.mean(maxps)), 6),
        "min_max_softmax": None if not maxps else round(float(np.min(maxps)), 6),
        "max_max_softmax": None if not maxps else round(float(np.max(maxps)), 6),
        "frac_max_softmax_ge_0.95": None if not maxps else round(float(np.mean(np.array(maxps) >= 0.95)), 6),
        "brier_multiclass": None if brier is None else round(brier, 6),
        "ece_10bin": ece,
        "ece_bins": bins,
        "metrics_source": "live_predict_raw_T1.0 on public_wound_dataset split",
        "note": "Accuracy is template recognition, not generalization, when unique_pixel_templates << n.",
    }


def probe(clf, name, group, img_rgb, source=None):
    quality = assess_input_quality(img_rgb)
    raw_t1 = clf.predict_raw(img_rgb, temperature=1.0)
    raw_t15 = clf.predict_raw(img_rgb, temperature=clf.temperature)
    gated = clf.predict(img_rgb)
    ordered = [c.capitalize() for c in clf.classes]
    raw_array_t1 = [float(raw_t1["probs"][c]) for c in ordered]
    return {
        "name": name,
        "group": group,
        "source": source,
        "shape": [int(img_rgb.shape[0]), int(img_rgb.shape[1]), int(img_rgb.shape[2])],
        "class_order": ordered,
        "raw_softmax_T1": {
            "array": [round(x, 6) for x in raw_array_t1],
            "probs": _round_probs(raw_t1["probs"]),
            "winner": raw_t1["winner"],
            "max_prob": round(float(raw_t1["max_prob"]), 6),
            "entropy": round(float(raw_t1["entropy"]), 6),
            "margin": round(float(raw_t1["margin"]), 6),
        },
        "raw_softmax_T1_5_temperature_scaled_not_a_gate": {
            "probs": _round_probs(raw_t15["probs"]),
            "winner": raw_t15["winner"],
            "max_prob": round(float(raw_t15["max_prob"]), 6),
        },
        "quality_gate": {
            "status": quality["status"],
            "reason": quality["reason"],
            "metrics": quality.get("metrics", {}),
        },
        "application_gated": {
            "status": gated.get("__status"),
            "reason": gated.get("__reason"),
            "winner": gated.get("__winner"),
            "is_confident": gated.get("__is_confident"),
            "class_probs_after_gate": {c: gated.get(c) for c in ordered},
        },
        "gate_hides_raw_injury_argmax": bool(
            quality["status"] != "VALID"
            and raw_t1["winner"] in {"Cut", "Bruise", "Swelling"}
            and float(raw_t1["max_prob"]) >= 0.80
        ),
    }


def load_rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def collect_probes(clf):
    rng = np.random.default_rng(42)
    out = []

    def add(name, group, img, source=None):
        out.append(probe(clf, name, group, img, source))

    add("gray", "uniform", np.full((224, 224, 3), 180, dtype=np.uint8), "generated_probe")
    add("black", "uniform", np.zeros((224, 224, 3), dtype=np.uint8), "generated_probe")
    add("white", "uniform", np.full((224, 224, 3), 255, dtype=np.uint8), "generated_probe")
    add("blank_midgray", "blank", np.full((300, 300, 3), 200, dtype=np.uint8), "generated_probe")

    blank = os.path.join("data", "datasets", "yolo_injury", "blank_skin.jpg")
    if os.path.isfile(blank):
        add("blank_skin.jpg", "no_injury_existing", load_rgb(blank), posix(blank))
    dummy = os.path.join("data", "datasets", "yolo_injury", "dummy_test.jpg")
    if os.path.isfile(dummy):
        add("dummy_test.jpg", "no_injury_existing", load_rgb(dummy), posix(dummy))
    add("skin_tone_uniform", "no_injury_generated", np.full((224, 224, 3), (185, 145, 125), dtype=np.uint8), "generated_probe")
    noisy_skin = np.clip(np.full((224, 224, 3), (185, 145, 125)) + rng.normal(0, 4, (224, 224, 3)), 0, 255).astype(np.uint8)
    add("skin_tone_noisy", "no_injury_generated", noisy_skin, "generated_probe")

    add("blue_unrelated", "unrelated_synthetic", np.full((224, 224, 3), (20, 60, 200), dtype=np.uint8), "generated_probe")
    add("green_unrelated", "unrelated_synthetic", np.full((224, 224, 3), (20, 180, 40), dtype=np.uint8), "generated_probe")
    striped = np.zeros((224, 224, 3), dtype=np.uint8)
    striped[:, :, 2] = 200
    striped[::6, :, 1] = 180
    add("blue_striped_unrelated", "unrelated_synthetic", striped, "generated_probe")
    add("high_frequency_noise", "unrelated_synthetic", rng.integers(0, 256, (224, 224, 3), dtype=np.uint8), "generated_probe")

    natural = [
        ("football_injury.jpg", "data/sample/image/football_injury.jpg", "unrelated_or_photo"),
        ("qa_swelling_offcenter.jpg", "data/sample/qa/qa_swelling_offcenter.jpg", "unrelated_or_photo"),
        ("upload_forensic_no_detection.jpg", "data/uploads/3e0dbd17-7475-487d-9f10-e7f9d6800238.jpg", "unrelated_or_photo"),
        ("upload_0047fcef.jpg", "data/uploads/0047fcef-5885-4467-a4d8-39e82247b167.jpg", "unrelated_or_photo"),
        ("synthetic_swollen_ankle.jpg", "data/uploads/synthetic_swollen_ankle.jpg", "unrelated_or_photo"),
    ]
    for name, rel, group in natural:
        if os.path.isfile(rel):
            rgb = load_rgb(rel)
            if rgb is not None:
                add(name, group, rgb, posix(rel))

    # Unique in-domain templates from the training set of the live checkpoint.
    seen = {}
    for row in csv.DictReader(open(MANIFEST, encoding="utf-8")):
        if row["split"] != "train":
            continue
        path = os.path.join(ROOT_DS, row["image_path"].replace("/", os.sep))
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        digest = _pixel_sha(bgr)
        if digest in seen:
            continue
        seen[digest] = row
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        add(
            f"train_unique_{row['class']}_{row['sample_id']}",
            "in_domain_unique_template",
            rgb,
            posix(path),
        )

    # Held-out test: one file per unique template, plus first two extra copies if any.
    test_seen = {}
    extras = 0
    for row in csv.DictReader(open(MANIFEST, encoding="utf-8")):
        if row["split"] != "test":
            continue
        path = os.path.join(ROOT_DS, row["image_path"].replace("/", os.sep))
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        digest = _pixel_sha(bgr)
        if digest not in test_seen:
            test_seen[digest] = row
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            add(
                f"test_unique_{row['class']}_{row['sample_id']}",
                "in_domain_heldout_template",
                rgb,
                posix(path),
            )
        elif extras < 2:
            extras += 1
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            add(
                f"test_duplicate_copy_{row['class']}_{row['sample_id']}",
                "in_domain_heldout_duplicate_copy",
                rgb,
                posix(path),
            )

    cut = np.full((224, 224, 3), (185, 145, 125), dtype=np.uint8)
    cv2.line(cut, (70, 40), (150, 180), (190, 20, 20), 7)
    add("synthetic_cut_template_redrawn", "in_domain_redrawn_generator", cut, "generated_same_as_download_public_datasets")
    add("blurred_cut_template", "blurred", cv2.GaussianBlur(cut, (31, 31), 8), "generated_probe")
    return out


def preprocessing_audit():
    return {
        "runtime_resize": "cv2.resize RGB to 224x224 (default interpolation INTER_LINEAR)",
        "runtime_scale": "uint8 / 255.0",
        "runtime_normalize": "ImageNet mean [0.485, 0.456, 0.406] std [0.229, 0.224, 0.225]",
        "channel_order": "RGB after cv2 BGR->RGB at callers; wrapper expects RGB",
        "training_augmentation": "NONE in WoundImageDataset (no flip, jitter, crop, mixup)",
        "training_freeze": "backbone frozen; only classifier/head trained",
        "training_loss": "CrossEntropyLoss class-weighted",
        "training_image_size": 224,
        "temperature_default": 1.5,
        "min_confidence_default": 0.80,
        "application_extra_gates": "max_prob>=0.80 AND margin>=0.20 AND entropy<=0.55*ln(C) AND input_quality VALID",
        "uniform_gate_reasons": [
            "uniform_or_blank_image",
            "near_uniform_palette",
            "blank_or_blurred_insufficient_detail",
            "low_contrast_blur",
            "high_frequency_unstructured",
            "colors_not_injury_like",
        ],
        "mismatch_note": "Training and inference both 224 + ImageNet norm. No train-time blank/OOD negatives for the promoted 3-class head.",
    }


def main():
    ident = artifact_identity()
    dataset = audit_dataset()
    clf = EfficientNetV2Classifier()
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(str(msg))

    log("=== EfficientNet forensic audit (no retrain) ===")
    log(f"canonical={ident['canonical_path']}")
    log(f"sha256={ident['sha256']}")
    log(f"n_classes={ident['n_classes_from_checkpoint']} wrapper_classes={list(clf.classes)}")
    log(f"metadata_sha_match={ident['sha_matches_metadata']} registry_sha_match={ident['sha_matches_registry']}")
    log(f"candidate_equals_production={ident['candidate_equals_production']}")
    log()
    log("=== Dataset ===")
    log(json.dumps({
        "class_balance": dataset["class_balance_all"],
        "splits": dataset["split_sizes"],
        "split_class": dataset["split_class_counts"],
        "unique_templates": dataset["unique_pixel_templates_by_class"],
        "cross_split_exact_groups": dataset["cross_split_exact_duplicate_group_count"],
        "subject_overlap": dataset["subject_overlap"],
        "negatives": dataset["negative_or_no_injury_labeled"],
    }, indent=2))

    test_eval = live_split_eval(clf, "test") if clf.is_loaded else None
    val_eval = live_split_eval(clf, "val") if clf.is_loaded else None
    log()
    log("=== Live held-out (raw T=1.0) ===")
    log(json.dumps({"test": test_eval, "val": val_eval}, indent=2))

    probes = collect_probes(clf) if clf.is_loaded else []
    log()
    log("=== RAW softmax T=1.0 complete arrays ===")
    log(f"class_order={[c.capitalize() for c in clf.classes]}")
    for row in probes:
        arr = row["raw_softmax_T1"]["array"]
        log(
            f"{row['name']:42s} group={row['group']:32s} "
            f"array={arr} winner={row['raw_softmax_T1']['winner']} "
            f"max={row['raw_softmax_T1']['max_prob']:.6f} "
            f"gate={row['quality_gate']['status']}/{row['application_gated']['status']} "
            f"app_winner={row['application_gated']['winner']}"
        )

    hidden = [p["name"] for p in probes if p["gate_hides_raw_injury_argmax"]]
    still_conf = [
        p["name"]
        for p in probes
        if p["group"] in {
            "uniform", "blank", "no_injury_existing", "no_injury_generated",
            "unrelated_synthetic", "unrelated_or_photo", "blurred",
        }
        and p["application_gated"].get("is_confident")
    ]
    log()
    log(f"gate_hides_raw_injury_argmax={hidden}")
    log(f"still_confident_on_ood_or_blank={still_conf}")

    gate_verdict = (
        "USEFUL_SAFETY_GUARD_HIDING_UNRELIABLE_CLASSIFIER"
        if hidden and not still_conf
        else "GATE_INCOMPLETE_OR_NOT_ONLY_HIDING"
    )
    if still_conf:
        gate_verdict = "GATE_LEAKS_CONFIDENT_OOD"
    elif hidden:
        gate_verdict = "USEFUL_SAFETY_GUARD_BUT_MODEL_STILL_UNRELIABLE"

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "did_not_retrain": True,
        "did_not_fabricate_metrics": True,
        "artifact": ident,
        "wrapper_classes": list(clf.classes),
        "dataset_audit": dataset,
        "leakage_audit": {
            "exact_pixel_duplicates_cross_split": dataset["cross_split_exact_duplicate_groups"],
            "subject_id_overlap": dataset["subject_overlap"],
            "subject_overlap_does_not_disprove_pixel_leakage": True,
            "near_duplicate_dhash": dataset["cross_split_near_duplicate_dhash_groups"],
            "unique_templates_total": dataset["unique_pixel_templates_total"],
            "source_leakage": "All splits come from the same PIL generator (download_public_datasets.py). Manifest 'source' is a false public-dataset claim.",
        },
        "preprocessing_audit": preprocessing_audit(),
        "live_test": test_eval,
        "live_val": val_eval,
        "probes": probes,
        "confidence_analysis": {
            "test_mean_max_softmax": (test_eval or {}).get("mean_max_softmax"),
            "test_frac_ge_0.95": (test_eval or {}).get("frac_max_softmax_ge_0.95"),
            "test_ece_10bin": (test_eval or {}).get("ece_10bin"),
            "test_brier": (test_eval or {}).get("brier_multiclass"),
            "ood_raw_typically_collapses_to_swelling": True,
        },
        "gate_analysis": {
            "verdict": gate_verdict,
            "uniform_gate_is": "useful_safety_guard",
            "uniform_gate_is_not": "a_fix_for_the_classifier",
            "names_where_gate_hides_raw_injury_argmax": hidden,
            "still_confident_on_ood_or_blank": still_conf,
            "do_not_remove_gates": True,
        },
        "root_cause": [
            "Production head was trained on ~5 unique PIL templates (cut/bruise/swelling plus remapped abrasion/laceration/burn drawings labeled swelling).",
            "Train/val/test share exact pixel duplicates, so 1.0 held-out accuracy is template memorization.",
            "Closed 3-class softmax has no negative/normal class, so blanks map to swelling at ~0.96-1.0.",
            "Head-only training with no augmentation and no OOD negatives.",
            "Quality gates withhold blanks; they do not make the classifier trustworthy.",
        ],
        "retraining_recommendation": [
            "Do not retrain on public_wound_dataset as-is.",
            "Need unique real photos, a true no-injury class, no silent remap, hash-disjoint splits, and OOD/blank labeled as reject/normal.",
            "A 4-class processed candidate already exists and was correctly NOT promoted (KEEP_BASELINE); it still collapsed some blanks to swelling.",
            "Keep current uniform/OOD gates until a verified replacement exists.",
        ],
        "final_status": "NOT_TRUSTWORTHY",
        "also": "REQUIRES_RETRAINING",
        "not": ["ALREADY_FIXED", "PARTIALLY_FIXED"],
        "held_out_accuracy_is_meaningful": False,
    }
    os.makedirs("scratch", exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with open(OUT_TXT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    log(f"wrote {OUT_JSON}")
    log(f"FINAL_STATUS={report['final_status']}")
    return report


if __name__ == "__main__":
    main()
