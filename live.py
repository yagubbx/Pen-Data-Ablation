"""Compare two models live and record a labeled clip for a fair replay test."""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from common import ROOT, models, panel, predict, save_json, set_threads, sha256

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=int, default=0, help='Webcam number')
    p.add_argument('--conf', type=float, default=0.45)
    p.add_argument('--device', default='cpu')
    p.add_argument('--condition', required=True, help='Describe the new room, light and background')
    p.add_argument('--novel', action='store_true', help='Confirm you reviewed dataset sheets and chose a new setup')
    p.add_argument('--output', type=Path, default=ROOT / 'captures')
    a = p.parse_args()
    if not 0 < a.conf <= 1:
        p.error('conf must be in (0, 1]')
    if not a.novel:
        p.error('Review reports/low_dataset.jpg and high_dataset.jpg, then use --novel for a new setup')
    set_threads()
    detectors = models()
    for model in detectors.values():
        predict(model, np.zeros((480, 640, 3), np.uint8), a.conf, a.device)
    cap = cv2.VideoCapture(a.source)
    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam. Try --source 1.')
    folder = a.output / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    folder.mkdir(parents=True)
    clip = folder / 'webcam.avi'
    writer = None
    frames = []
    present = None
    times = []
    start = time.perf_counter()
    positives = negatives = 0
    print('P: pen present | N: no pen | U: pause labels | Q: finish')
    print('Press U before changing the scene. Only P/N frames are saved.')
    try:
        while True:
            loop = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'MJPG'), 10, (w, h))
                if not writer.isOpened():
                    raise RuntimeError('Cannot save webcam clip')
            views = []
            for name, model in detectors.items():
                boxes = predict(model, frame, a.conf, a.device)
                views.append(panel(frame, boxes, name.upper()))
            if present is not None:
                writer.write(frame)
                frames.append({'index': len(frames), 'present': present, 'condition': a.condition, 'seconds': round(time.perf_counter() - start, 4)})
                positives += int(present)
                negatives += int(not present)
            side = np.hstack(views)
            fps = len(times) / sum(times) if times else 0
            state = 'PAUSED' if present is None else 'PEN PRESENT' if present else 'NO PEN'
            cv2.rectangle(side, (0, 440), (1280, 480), (20, 20, 20), -1)
            text = f'{state} | P {positives} / N {negatives} frames | Both-model loop {fps:.1f} FPS | P/N/U/Q'
            cv2.putText(side, text, (10, 467), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow('Low diversity | High diversity', side)
            key = cv2.waitKey(1) & 255
            times.append(time.perf_counter() - loop)
            times = times[-30:]
            if key == ord('q'):
                break
            if key == ord('p'):
                present = True
            if key == ord('n'):
                present = False
            if key == ord('u'):
                present = None
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        if frames and clip.exists():
            save_json(clip.with_suffix('.json'), {'source': 'webcam', 'live_camera': True, 'novel_condition_confirmed': a.novel, 'condition_note': a.condition, 'label_type': 'presence', 'video_sha256': sha256(clip), 'frames': frames, 'recorded_confidence': a.conf, 'video_fps': 10, 'playback_note': 'Fixed 10 FPS playback. Actual capture times are in frames[].seconds.', 'weights': {n: sha256(ROOT / f'weights/{n}.pt') for n in detectors}})
            print(f'Saved {len(frames)} labeled frames: {clip}')
            print(f'Run: python compare.py --clip "{clip}" --conf {a.conf}')
        else:
            print('No labeled frames saved. Press P or N while recording.')
    if frames:
        print('Comparing the saved frames. Please wait...')
        subprocess.run([sys.executable, str(ROOT / 'compare.py'), '--clip', str(clip), '--conf', str(a.conf), '--device', a.device, '--output', str(folder / 'comparison')], check=True)
if __name__ == '__main__':
    main()
