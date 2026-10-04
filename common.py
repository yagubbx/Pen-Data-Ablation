"""Shared settings, dataset checks and object matching."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'runs/settings'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runs/matplotlib'))
for key in ('YOLO_CONFIG_DIR', 'MPLCONFIGDIR'):
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def set_threads(count=2):
    import torch
    from ultralytics.utils import torch_utils
    torch_utils.NUM_THREADS = count
    torch.set_num_threads(count)


def dataset():
    import cv2
    import math
    splits = json.loads((ROOT / 'data/splits.json').read_text())
    rows = json.loads((ROOT / 'data/manifest.json').read_text())
    assert set(splits['low']) <= set(splits['high'])
    groups, hashes, counts = {}, set(), {}
    for name in ('high', 'val', 'test'):
        folder = 'train' if name == 'high' else name
        counts[name] = len(splits[name])
        for filename in splits[name]:
            image = ROOT / 'data/images' / folder / filename
            label = ROOT / 'data/labels' / folder / Path(filename).with_suffix('.txt')
            assert cv2.imread(str(image)) is not None, image
            checksum = sha256(image)
            assert checksum not in hashes, f'Duplicate image: {filename}'
            hashes.add(checksum)
            row = next(r for r in rows if r['filename'] == filename)
            assert checksum == row['sha256'], filename
            author = row['attribution']['AuthorProfileURL']
            assert groups.get(author, name) == name, 'Author appears in different splits'
            groups[author] = name
            assert label.is_file(), label
            for line in label.read_text().splitlines():
                c, x, y, w, h = map(float, line.split())
                assert all(math.isfinite(v) for v in (c, x, y, w, h))
                assert c == 0 and 0 < w <= 1 and 0 < h <= 1
                assert x-w/2 >= -1e-5 and x+w/2 <= 1.00001
                assert y-h/2 >= -1e-5 and y+h/2 <= 1.00001
    counts['low'] = len(splits['low'])
    return splits, counts


def iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    total = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap / max(total, 1e-9)


def match(predictions, truth):
    """Greedy confidence-ordered one-to-one matching, IoU >= 0.5."""
    used, tp, fp = set(), 0, 0
    for box in sorted(predictions, key=lambda b: b[4], reverse=True):
        candidates = [(iou(box[:4], gt), i) for i, gt in enumerate(truth) if i not in used]
        score, index = max(candidates, default=(0, -1))
        if score >= 0.5:
            used.add(index)
            tp += 1
        else:
            fp += 1
    return {'tp': tp, 'fp': fp, 'fn': len(truth)-len(used)}


def metrics(counts):
    tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
    return {**counts, 'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None}


def boxes_for(filename, split, width, height):
    path = ROOT / 'data/labels' / split / Path(filename).with_suffix('.txt')
    result = []
    for line in path.read_text().splitlines():
        _, x, y, w, h = map(float, line.split())
        result.append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
    return result


def models():
    from ultralytics import YOLO
    result = {}
    for name in ('low', 'high'):
        path = ROOT / 'weights' / f'{name}.pt'
        if not path.is_file():
            raise FileNotFoundError('Run train.py first. Missing: ' + str(path))
        model = YOLO(str(path))
        if model.names != {0: 'pen'}:
            raise ValueError(f'Wrong classes: {path}')
        result[name] = model
    return result


def predict(model, frame, conf=0.45, device='cpu'):
    return model.predict(frame, imgsz=416, conf=conf, iou=0.45, device=device,
                         verbose=False)[0].boxes.data.cpu().tolist()


def panel(frame, boxes, title, truth=None):
    import cv2
    import numpy as np
    height, width = frame.shape[:2]
    scale = min(640 / width, 452 / height)
    w, h = round(width * scale), round(height * scale)
    left, top = (640-w)//2, 28+(452-h)//2
    out = np.full((480, 640, 3), 25, np.uint8)
    out[top:top+h, left:left+w] = cv2.resize(frame, (w, h))

    def point(x, y):
        return round(x*w/width)+left, round(y*h/height)+top

    for x1, y1, x2, y2 in truth or []:
        cv2.rectangle(out, point(x1,y1), point(x2,y2), (255,180,0), 1)
    for x1, y1, x2, y2, confidence, *_ in boxes:
        start, end = point(x1,y1), point(x2,y2)
        cv2.rectangle(out, start, end, (0, 220, 0), 2)
        cv2.putText(out, f'pen {confidence:.2f}', (max(0, start[0]), max(42, start[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (25, 25, 25), -1)
    cv2.putText(out, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out
