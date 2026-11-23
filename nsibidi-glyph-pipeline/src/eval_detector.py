"""
Evaluate a trained glyph detector (best.pth from train_rcnn_local.py) on a composed-pages folder.

Reports, on the VALIDATION pages:
  * mAP@0.5 (should reproduce the 'val_map50' stored in best.pth if you point it at the folder the
    detector was trained on -- this is how you confirm which folder was used)
  * AP@0.75 and AP@[0.5:0.95] (stricter box-fit measures)
  * precision / recall / F1 at a score threshold (default 0.5, IoU 0.5)
Single-class ('glyph') detector. Self-contained (does not import train_rcnn_local.py).

  python eval_detector.py --checkpoint best.pth --pages-dir "<ROOT>\\nsibidi_training_data\\detection_pages"
"""
import argparse, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn_v2)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

VAL_FRACTION, SEED = 0.1, 42


class _T:  # small namespace so the rest of the script reads the same
    pass


T = _T()
T.VAL_FRACTION, T.SEED = VAL_FRACTION, SEED


class CocoPages(Dataset):
    def __init__(self, images_dir, coco_json_path, image_ids=None, single_class=True):
        self.images_dir, self.single_class = images_dir, single_class
        coco = json.load(open(coco_json_path))
        self.images_by_id = {im["id"]: im for im in coco["images"]}
        self.anns = {}
        for a in coco["annotations"]:
            self.anns.setdefault(a["image_id"], []).append(a)
        self.image_ids = image_ids if image_ids is not None else sorted(self.images_by_id)

    def ids_for_split(self, split):
        return [i for i in sorted(self.images_by_id) if self.images_by_id[i].get("split") == split]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        iid = self.image_ids[idx]
        info = self.images_by_id[iid]
        img = Image.open(os.path.join(self.images_dir, info["file_name"])).convert("RGB")
        boxes = []
        for a in self.anns.get(iid, []):
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
        bt = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        return TF.to_tensor(img), {"boxes": bt}


def _build_model(arch, num_classes, min_size, max_size):
    f = fasterrcnn_mobilenet_v3_large_fpn if arch == "mobilenet" else fasterrcnn_resnet50_fpn_v2
    m = f(weights=None, weights_backbone=None, min_size=min_size, max_size=max_size)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, num_classes)
    return m


def _collate(b):
    return tuple(zip(*b))


def _average_precision(records, n_gt):
    if n_gt == 0:
        return None
    if not records:
        return 0.0
    records = sorted(records, key=lambda r: -r[0])
    tp = np.cumsum([1.0 if r[1] else 0.0 for r in records])
    fp = np.cumsum([0.0 if r[1] else 1.0 for r in records])
    rec, prec = tp / n_gt, tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


T._average_precision = _average_precision
T.CocoDetectionDataset = CocoPages
T.build_model = lambda arch, nc, pretrained, mn, mx: _build_model(arch, nc, mn, mx)
T.collate_fn = _collate


def ap_at(preds, gts, iou_t):
    """preds: list of (boxes[N,4], scores[N]) per page; gts: list of boxes[M,4] per page."""
    records, n_gt = [], 0
    for (pb, ps), g in zip(preds, gts):
        n_gt += len(g)
        order = ps.argsort(descending=True)
        pb, ps = pb[order], ps[order]
        matched = torch.zeros(len(g), dtype=torch.bool)
        ious = box_iou(pb, g) if len(pb) and len(g) else None
        for i in range(len(pb)):
            tp = False
            if ious is not None:
                best, j = ious[i].max(0)
                if best >= iou_t and not matched[j]:
                    matched[j] = True
                    tp = True
            records.append((ps[i].item(), tp))
    return T._average_precision(records, n_gt)


def prf(preds, gts, score_t, iou_t=0.5):
    tp = fp = fn = 0
    for (pb, ps), g in zip(preds, gts):
        keep = ps >= score_t
        pb, ps = pb[keep], ps[keep]
        order = ps.argsort(descending=True)
        pb = pb[order]
        matched = torch.zeros(len(g), dtype=torch.bool)
        ious = box_iou(pb, g) if len(pb) and len(g) else None
        for i in range(len(pb)):
            ok = False
            if ious is not None:
                best, j = ious[i].max(0)
                if best >= iou_t and not matched[j]:
                    matched[j] = True
                    ok = True
            tp += ok
            fp += (not ok)
        fn += int((~matched).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9), tp, fp, fn


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pages-dir", required=True)
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=2)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    meta = ck["meta"]
    model = T.build_model(meta["arch"], meta["num_classes"], False, meta["min_size"], meta["max_size"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"checkpoint: epoch {ck.get('epoch')}  stored val_map50 {ck.get('val_map50')}")

    images_dir = os.path.join(a.pages_dir, "images")
    ann = os.path.join(a.pages_dir, "annotations_coco.json")
    probe = T.CocoDetectionDataset(images_dir, ann, single_class=True)
    val_ids = probe.ids_for_split("val")
    train_ids = probe.ids_for_split("train")
    if val_ids and train_ids:
        print(f"split: stored glyph-file split  (train {len(train_ids)} pages, val {len(val_ids)} pages)")
    else:
        ids = probe.image_ids
        n_val = max(1, int(len(ids) * T.VAL_FRACTION))
        perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(T.SEED)).tolist()
        val_ids = [ids[i] for i in perm[:n_val]]
        print(f"split: NO stored split -> random 10% of pages (seed {T.SEED}), {len(val_ids)} val pages")

    ds = T.CocoDetectionDataset(images_dir, ann, val_ids, single_class=True)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, collate_fn=T.collate_fn)
    preds, gts = [], []
    for k, (images, targets) in enumerate(loader):
        outs = model([im.to(device) for im in images])
        for o, t in zip(outs, targets):
            preds.append((o["boxes"].cpu(), o["scores"].cpu()))
            gts.append(t["boxes"])
        print(f"  {min((k + 1) * a.batch_size, len(ds))}/{len(ds)} pages", end="\r")
    print()

    ap50 = ap_at(preds, gts, 0.5)
    ap75 = ap_at(preds, gts, 0.75)
    aps = [ap_at(preds, gts, t) for t in np.arange(0.5, 0.96, 0.05)]
    p, r, f1, tp, fp, fn = prf(preds, gts, a.score)
    print(f"\nValidation pages: {len(ds)}   ground-truth boxes: {sum(len(g) for g in gts)}")
    print(f"AP@0.50        : {ap50:.4f}")
    print(f"AP@0.75        : {ap75:.4f}")
    print(f"AP@[0.50:0.95] : {float(np.mean(aps)):.4f}")
    print(f"At score >= {a.score}, IoU 0.5: precision {p:.3f}  recall {r:.3f}  F1 {f1:.3f}   (TP {tp}, FP {fp}, FN {fn})")


if __name__ == "__main__":
    main()
