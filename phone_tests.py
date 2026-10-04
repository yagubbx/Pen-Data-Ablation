"""Prepare labeled phone clips and compare both models on the same frames."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
from common import ROOT, save_json, sha256


def prepare(spec, sample_fps):
    source = ROOT / 'videos' / spec['file']
    if spec.get('source_sha256') and sha256(source) != spec['source_sha256']:
        raise ValueError(f'{source.name} changed. Review its time labels before running again.')
    segments = spec['segments']
    targets = []
    previous_end = 0
    for start, end, present in segments:
        if start < previous_end or end <= start or type(present) is not bool:
            raise ValueError('Segments must be ordered, separate, and have true/false labels.')
        for step in range(math.ceil((end-start)*sample_fps)):
            seconds = start + step/sample_fps
            if seconds < end:
                targets.append((seconds, end, present))
        previous_end = end
    if not targets:
        raise ValueError('No selected frames.')
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {source.name}')
    output = ROOT / 'test_clip/phone' / (source.stem + '.avi')
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frames = []
    source_index = 0
    last_time = -1
    try:
        while len(frames) < len(targets):
            ok, frame = cap.read()
            if not ok:
                break
            seconds = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if seconds < last_time:
                raise ValueError('Video timestamps moved backwards.')
            last_time = seconds
            target, end, present = targets[len(frames)]
            if seconds + 1e-6 >= target:
                if seconds >= end or seconds-target > 1/sample_fps:
                    raise ValueError('Missing video frames near a selected time.')
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'MJPG'),
                                             sample_fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError('Cannot save the prepared phone clip.')
                writer.write(frame)
                frames.append({'index': len(frames), 'source_frame': source_index,
                               'source_seconds': round(seconds, 6), 'present': present,
                               'condition': spec['condition']})
            source_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if len(frames) != len(targets):
        raise ValueError(f'{source.name}: expected {len(targets)} frames, got {len(frames)}.')
    save_json(output.with_suffix('.json'), {
        'source': 'recorded phone video; not live webcam inference',
        'live_camera': False, 'novel_condition_confirmed': False,
        'condition_note': spec['condition'], 'label_type': 'presence',
        'source_file': 'videos/' + source.name, 'source_sha256': sha256(source),
        'video_sha256': sha256(output), 'sample_fps': sample_fps,
        'segments': segments,
        'label_note': 'Presence labels from visual review. Transition gaps excluded before prediction.',
        'frames': frames})
    print(f'{source.name}: {len(frames)} selected frames', flush=True)
    return output


def summary(specs):
    lines = ['# Phone Video Results', '',
             'Three phone recordings were tested offline. See LIVE_RESULTS.md for the separate live test.',
             'Confidence: 0.45. Input: 416. No smoothing. Both models read the same saved frames.',
             'Five frames per second were selected in visually labeled intervals.',
             'Unclear transitions were excluded before looking at predictions.', '',
             '| Video | Model | Positive / negative frames | TP | FP | FN | TN | Presence precision | Presence recall |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    records = {}
    for spec in specs:
        name = Path(spec['file']).stem
        record = json.loads((ROOT / f'reports/phone/{name}/metrics.json').read_text())
        records[name] = record
        coverage = record['coverage']
        for model, counts in record['results'].items():
            percent = lambda value: 'N/A' if value is None else f'{100*value:.1f}%'
            lines.append(f'| {name} | {model} | {coverage["positive_frames"]} / {coverage["negative_frames"]} '
                         f'| {counts["tp"]} | {counts["fp"]} | {counts["fn"]} | {counts["tn"]} '
                         f'| {percent(counts["precision"])} | {percent(counts["recall"])} |')
    lines += ['', 'FP: a box when no pen is visible. FN: no box when a pen is visible.',
              'These are frame-level presence counts, not box-location scores.',
              'A wrong box in a pen-present frame can still count as a presence hit.',
              'A 100% presence precision does not mean all boxes are correct.',
              'See [PHONE_REVIEW.md](PHONE_REVIEW.md) for visual checks of the included run.',
              'Frames from one video are related, so these are not independent examples.', '',
              '## Files', '',
              'Originals: `videos/`. Reviewed intervals: `videos/segments.json`.',
              'Prepared clips and per-frame labels: `test_clip/phone/`.',
              'Reports, screenshots and side-by-side videos: `reports/phone/`.', '',
              'The original phone files are unchanged. The prepared clips use MJPEG at 5 FPS.',
              'The supplied phone files are 478 x 850 pixels. Source timestamps are saved in the labels.',
              'These new recordings were not used to train either model. Dim and bright clips share',
              'a surface, objects and pen, so they are a small lighting comparison, not separate rooms.', '',
              '## Next data to collect', '',
              'Use the missed views to collect more thin light pens on dark surfaces, small pens,',
              'and clutter under different lighting. Include empty scenes. Label pen boxes, train',
              'again, and test on a new recording. Keep these videos out of training if they remain the test set.', '',
              'For the live requirement, run `live.py` with a webcam as described in START.md.']
    (ROOT / 'PHONE_RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    save_json(ROOT / 'reports/phone/summary.json', records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT / 'videos/segments.json').read_text())
    rate = config['sample_fps']
    if not isinstance(rate, (int, float)) or not 0 < rate <= 30:
        raise ValueError('sample_fps must be between 0 and 30.')
    for spec in config['clips']:
        clip = prepare(spec, rate)
        if not args.prepare_only:
            subprocess.run([sys.executable, str(ROOT / 'compare.py'), '--clip', str(clip),
                            '--output', str(ROOT / 'reports/phone' / clip.stem)], check=True)
    if not args.prepare_only:
        summary(config['clips'])
        print('Done. Open PHONE_RESULTS.md and reports/phone/.')


if __name__ == '__main__':
    main()
