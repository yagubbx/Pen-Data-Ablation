"""Evaluate both models on identical held-out images and training conditions."""
import argparse
import csv
import time
import cv2
import numpy as np
from common import ROOT, boxes_for, dataset, match, metrics, models, panel, predict, save_json, set_threads, sha256

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--conf', type=float, default=0.45)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    if not 0 < a.conf <= 1:
        p.error('conf must be in (0, 1]')
    splits, _ = dataset()
    set_threads()
    detectors = models()
    out = ROOT / 'reports/evaluation'
    out.mkdir(parents=True, exist_ok=True)
    totals = {}
    rows = []
    groups = {'test': splits['test'], 'low_train': splits['low'], 'high_only_train': [f for f in splits['high'] if f not in splits['low']]}
    for group, filenames in groups.items():
        split = 'test' if group == 'test' else 'train'
        counts = {name: dict(tp=0, fp=0, fn=0) for name in detectors}
        for filename in filenames:
            frame = cv2.imread(str(ROOT / 'data/images' / split / filename))
            h, w = frame.shape[:2]
            truth = boxes_for(filename, split, w, h)
            panels = []
            for name, model in detectors.items():
                start = time.perf_counter()
                boxes = predict(model, frame, a.conf, a.device)
                elapsed = time.perf_counter() - start
                result = match(boxes, truth)
                for k, v in result.items():
                    counts[name][k] += v
                rows.append({'group': group, 'file': filename, 'model': name, **result, 'inference_ms': round(elapsed * 1000, 2)})
                if group == 'test':
                    view = panel(frame, boxes, f"{name.upper()} | TP {result['tp']} FP {result['fp']} FN {result['fn']}", truth)
                    panels.append(view)
            if panels:
                examples = out / 'examples'
                examples.mkdir(exist_ok=True)
                cv2.imwrite(str(examples / filename), np.hstack(panels))
        totals[group] = {name: metrics(c) for name, c in counts.items()}
    with (out / 'per_image.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record = {'confidence': a.conf, 'iou_match': 0.5, 'nms_iou': 0.45, 'imgsz': 416, 'smoothing': False, 'weights': {n: sha256(ROOT / f'weights/{n}.pt') for n in detectors}, 'splits_sha256': sha256(ROOT / 'data/splits.json'), 'results': totals, 'note': 'Only test is held out. Training-condition scores are not generalization estimates.'}
    save_json(out / 'metrics.json', record)
    for group, result in totals.items():
        print(group, result)
if __name__ == '__main__':
    main()
