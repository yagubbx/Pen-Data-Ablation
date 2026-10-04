"""Count errors for both models on every frame of the same saved clip."""
import argparse
import csv
import json
import math
import time
from pathlib import Path
import cv2
import numpy as np
from common import ROOT, match, metrics, models, panel, predict, save_json, set_threads, sha256

def validate_labels(info):
    kind = info.get('label_type')
    if kind not in ('boxes', 'presence'):
        raise ValueError('label_type must be boxes or presence')
    frames = info.get('frames', [])
    if not frames:
        raise ValueError('No labeled frames. Record some P and N segments first.')
    for i, row in enumerate(frames):
        if row.get('index') != i:
            raise ValueError('Frame labels must be consecutive and start at 0')
        if kind == 'presence' and type(row.get('present')) is not bool:
            raise ValueError('Each frame needs a true/false present label')
        if kind == 'boxes':
            if not isinstance(row.get('boxes'), list):
                raise ValueError('Each frame needs a boxes list')
            for b in row['boxes']:
                if len(b) != 4 or not all((math.isfinite(v) for v in b)) or b[2] <= b[0] or (b[3] <= b[1]):
                    raise ValueError('Invalid ground-truth box')
    return (kind, frames)

def score(boxes, truth, kind):
    if kind == 'boxes':
        return match(boxes, truth['boxes'])
    found = bool(boxes)
    present = truth['present']
    return {'tp': int(found and present), 'fp': int(found and (not present)), 'fn': int(not found and present), 'tn': int(not found and (not present))}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clip', type=Path, default=ROOT / 'test_clip/photos.avi')
    p.add_argument('--labels', type=Path, help='Default: same path as clip, with .json')
    p.add_argument('--conf', type=float, default=0.45)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if not 0 < a.conf <= 1:
        p.error('conf must be in (0, 1]')
    labels = a.labels or a.clip.with_suffix('.json')
    info = json.loads(labels.read_text(encoding='utf-8'))
    kind, truth = validate_labels(info)
    if sha256(a.clip) != info.get('video_sha256'):
        p.error('Clip hash does not match the labels')
    out = a.output or ROOT / 'reports' / f'clip_{a.clip.stem}'
    out.mkdir(parents=True, exist_ok=True)
    # Keep the original clip safe.
    video = out / 'side_by_side.avi'
    if video.resolve() == a.clip.resolve():
        p.error('Output must not replace the input clip')
    set_threads()
    detectors = models()
    cap = cv2.VideoCapture(str(a.clip))
    if not cap.isOpened():
        raise RuntimeError('Cannot open clip')
    width, height = (int(cap.get(3)), int(cap.get(4)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), fps, (1280, 480))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError('Cannot save comparison video')
    keys = ('tp', 'fp', 'fn', 'tn') if kind == 'presence' else ('tp', 'fp', 'fn')
    totals = {n: dict.fromkeys(keys, 0) for n in detectors}
    timings = {n: [] for n in detectors}
    by_condition = {}
    rows = []
    i = 0
    screenshots = 0
    try:
        # Warm up both models before timing.
        warm = np.zeros((height, width, 3), np.uint8)
        for model in detectors.values():
            predict(model, warm, a.conf, a.device)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i >= len(truth):
                raise ValueError('Clip has more frames than labels')
            row = truth[i]
            condition = row.get('condition', 'unspecified')
            grouped = by_condition.setdefault(condition, {n: dict.fromkeys(keys, 0) for n in detectors})
            views = []
            predicted = {}
            for name, model in detectors.items():
                start = time.perf_counter()
                boxes = predict(model, frame, a.conf, a.device)
                elapsed = time.perf_counter() - start
                timings[name].append(elapsed)
                count = score(boxes, row, kind)
                predicted[name] = count
                for k, v in count.items():
                    totals[name][k] += v
                    grouped[name][k] += v
                rows.append({'frame': i, 'condition': condition, 'model': name, **count, 'boxes': len(boxes), 'inference_ms': round(elapsed * 1000, 2)})
                title = f"{name.upper()} | frame {i} | FP {count['fp']} FN {count['fn']}"
                view = panel(frame, boxes, title, row['boxes'] if kind == 'boxes' else None)
                views.append(view)
            side = np.hstack(views)
            writer.write(side)
            if screenshots < 6 and (predicted['low'] != predicted['high'] or i == 0):
                cv2.imwrite(str(out / f'frame_{i}.jpg'), side)
                screenshots += 1
            i += 1
    finally:
        cap.release()
        writer.release()
    if i != len(truth):
        raise ValueError(f'Clip has {i} frames but labels have {len(truth)}')
    with (out / 'per_frame.csv').open('w', newline='', encoding='utf-8') as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    counts = {n: metrics(c) for n, c in totals.items()}
    result = {'source': info.get('source'), 'live_camera': info.get('live_camera', False), 'novel_condition_confirmed': info.get('novel_condition_confirmed', False), 'condition_note': info.get('condition_note'), 'frames': i, 'label_type': kind, 'count_unit': 'objects at IoU >= 0.5' if kind == 'boxes' else 'frames with/without any pen', 'confidence': a.conf, 'smoothing': False, 'imgsz': 416, 'nms_iou': 0.45, 'clip_sha256': sha256(a.clip), 'labels_sha256': sha256(labels), 'weights': {n: sha256(ROOT / f'weights/{n}.pt') for n in detectors}, 'results': counts, 'by_condition': {c: {n: metrics(v) for n, v in group.items()} for c, group in by_condition.items()}, 'inference_ms_mean': {n: 1000 * sum(t) / len(t) for n, t in timings.items()}}
    if kind == 'presence':
        result['coverage'] = {'positive_frames': sum((r['present'] for r in truth)), 'negative_frames': sum((not r['present'] for r in truth))}
        result['coverage_complete'] = all(result['coverage'].values())
    save_json(out / 'metrics.json', result)
    lines = ['# Clip Comparison', '', f'Frames: {i}. Confidence: {a.conf}.', f"Count unit: {result['count_unit']}.", '', '| Model | TP | FP | FN | Precision | Recall |', '|---|---:|---:|---:|---:|---:|']
    percent = lambda v: 'N/A' if v is None else f'{100 * v:.1f}%'
    for n, c in counts.items():
        lines.append(f"| {n} | {c['tp']} | {c['fp']} | {c['fn']} | {percent(c['precision'])} | {percent(c['recall'])} |")
    lines += ['', 'See side_by_side.avi, frame images and per_frame.csv.', 'Green boxes are predictions. Blue boxes, when shown, are labels.', 'Presence scores do not measure box location or extra boxes in positive frames.' if kind == 'presence' else 'This photo clip is an offline test. It does not prove live webcam performance.']
    lines += ['', f"Low missed {counts['low']['fn']}; high missed {counts['high']['fn']}.", f"Low had {counts['low']['fp']} false positives; high had {counts['high']['fp']}.", 'Check the saved frames to see which backgrounds or pen positions caused errors.', 'Collect more examples of those cases, plus empty scenes. Test on a new clip after retraining.']
    if kind == 'presence' and (not result['coverage_complete']):
        lines += ['', 'Incomplete test: record both pen-present and pen-absent segments.']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(counts, indent=2))
    print(f'Report: {out}')
if __name__ == '__main__':
    main()
