import glob
import os

import numpy as np

# ============================================================
# Paths
# ============================================================

LABEL_DIR = "/home/pi/datasets/test/labels"
PRED_DIR = "/home/pi/meld-yolo/test_results_fp32"

NUM_CLASSES = 4

# IoU thresholds for mAP50-95
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)


# ============================================================
# Utilities
# ============================================================


def load_gt(txt_path):
    """YOLO label: class cx cy w h normalized [0,1].

    Convert to pixel coordinates assuming 640x640.
    """
    gts = []

    if not os.path.exists(txt_path):
        return gts

    with open(txt_path) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            cx = float(parts[1]) * 640.0
            cy = float(parts[2]) * 640.0
            w = float(parts[3]) * 640.0
            h = float(parts[4]) * 640.0

            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0

            gts.append({"class": cls, "box": [x1, y1, x2, y2]})

    return gts


def load_pred(txt_path):
    """Prediction: class confidence x1 y1 x2 y2.
    """
    preds = []

    if not os.path.exists(txt_path):
        return preds

    with open(txt_path) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()
            if len(parts) < 6:
                continue

            cls = int(float(parts[0]))
            score = float(parts[1])

            x1 = float(parts[2])
            y1 = float(parts[3])
            x2 = float(parts[4])
            y2 = float(parts[5])

            preds.append({"class": cls, "score": score, "box": [x1, y1, x2, y2]})

    return preds


def box_iou(box1, box2):
    """Box = [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)

    inter = iw * ih

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])

    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter

    if union <= 0.0:
        return 0.0

    return inter / union


def compute_ap(recall, precision):
    """COCO-style 101-point interpolated AP."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    x = np.linspace(0.0, 1.0, 101)

    ap = 0.0

    for r in x:
        inds = np.where(mrec >= r)[0]

        if len(inds):
            ap += np.max(mpre[inds])

    return ap / 101.0


# ============================================================
# Collect image names
# ============================================================

label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.txt")))

if not label_files:
    raise RuntimeError("No label files found.")

print("=" * 60)
print(" YOLO26 FP32 Test Set Evaluation")
print("=" * 60)

print("Labels     :", LABEL_DIR)
print("Predictions:", PRED_DIR)
print("Images     :", len(label_files))
print("Classes    :", NUM_CLASSES)
print()


# ============================================================
# Load whole dataset
# ============================================================

dataset = []

for label_path in label_files:
    name = os.path.splitext(os.path.basename(label_path))[0]

    pred_path = os.path.join(PRED_DIR, name + ".txt")

    gts = load_gt(label_path)
    preds = load_pred(pred_path)

    dataset.append({"name": name, "gts": gts, "preds": preds})


# ============================================================
# Evaluation
# ============================================================

all_ap = np.zeros((NUM_CLASSES, len(IOU_THRESHOLDS)))

total_tp_50 = 0
total_fp_50 = 0
total_fn_50 = 0

class_p = []
class_r = []

for cls in range(NUM_CLASSES):
    cls_aps = []

    for ti, iou_thr in enumerate(IOU_THRESHOLDS):
        # ----------------------------------------------------
        # Gather all predictions of this class
        # ----------------------------------------------------

        records = []

        total_gt = 0

        for item in dataset:
            gts_cls = [g for g in item["gts"] if g["class"] == cls]

            preds_cls = [p for p in item["preds"] if p["class"] == cls]

            total_gt += len(gts_cls)

            for p in preds_cls:
                records.append({"score": p["score"], "box": p["box"], "image": item["name"], "matched": False})

        # Sort confidence descending
        records.sort(key=lambda x: x["score"], reverse=True)

        tp = np.zeros(len(records))
        fp = np.zeros(len(records))

        # ----------------------------------------------------
        # Match predictions
        # ----------------------------------------------------

        gt_used = {}

        for i, rec in enumerate(records):
            image_name = rec["image"]

            item = next(x for x in dataset if x["name"] == image_name)

            gts_cls = [g for g in item["gts"] if g["class"] == cls]

            if image_name not in gt_used:
                gt_used[image_name] = np.zeros(len(gts_cls), dtype=bool)

            used = gt_used[image_name]

            best_iou = iou_thr
            best_idx = -1

            for gi, gt in enumerate(gts_cls):
                if used[gi]:
                    continue

                iou = box_iou(rec["box"], gt["box"])

                if iou >= best_iou:
                    best_iou = iou
                    best_idx = gi

            if best_idx >= 0:
                tp[i] = 1
                used[best_idx] = True
            else:
                fp[i] = 1

        # ----------------------------------------------------
        # AP
        # ----------------------------------------------------

        if total_gt == 0 or len(records) == 0:
            ap = 0.0

        else:
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)

            recall = tp_cum / max(total_gt, 1)

            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

            ap = compute_ap(recall, precision)

        all_ap[cls, ti] = ap
        cls_aps.append(ap)

        # ----------------------------------------------------
        # P / R at IoU=0.50
        # ----------------------------------------------------

        if abs(iou_thr - 0.50) < 1e-6:
            total_tp_50 += int(np.sum(tp))
            total_fp_50 += int(np.sum(fp))
            total_fn_50 += total_gt - int(np.sum(tp))

            if total_gt > 0:
                class_r.append(np.sum(tp) / total_gt)
            else:
                class_r.append(0.0)

            denom = np.sum(tp) + np.sum(fp)

            if denom > 0:
                class_p.append(np.sum(tp) / denom)
            else:
                class_p.append(0.0)


# ============================================================
# Final metrics
# ============================================================

mAP50_per_class = all_ap[:, 0]
mAP95_per_class = np.mean(all_ap, axis=1)

mAP50 = np.mean(mAP50_per_class)
mAP5095 = np.mean(mAP95_per_class)

precision = total_tp_50 / max(total_tp_50 + total_fp_50, 1)

recall = total_tp_50 / max(total_tp_50 + total_fn_50, 1)


# ============================================================
# Output
# ============================================================

print("=" * 60)
print(" Overall Metrics")
print("=" * 60)

print(f"Precision       : {precision:.6f}")

print(f"Recall          : {recall:.6f}")

print(f"mAP50           : {mAP50:.6f}")

print(f"mAP50-95        : {mAP5095:.6f}")

print()

print("=" * 60)
print(" Per-Class Metrics")
print("=" * 60)

for cls in range(NUM_CLASSES):
    print(
        f"Class {cls}: "
        f"P={class_p[cls]:.6f}, "
        f"R={class_r[cls]:.6f}, "
        f"AP50={mAP50_per_class[cls]:.6f}, "
        f"AP50-95={mAP95_per_class[cls]:.6f}"
    )

print()

print("=" * 60)
print(" Detection Statistics")
print("=" * 60)

print("TP @ IoU=0.50 :", total_tp_50)
print("FP @ IoU=0.50 :", total_fp_50)
print("FN @ IoU=0.50 :", total_fn_50)

print()
print("Evaluation finished.")
