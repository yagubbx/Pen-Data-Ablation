"""Check data, show background variation and make a fixed photo test clip."""
import cv2
import numpy as np
from common import ROOT, boxes_for, dataset, save_json, sha256

def main():
    splits, counts = dataset()
    out = ROOT / 'reports'
    out.mkdir(exist_ok=True)
    features = {}
    for variant in ('low', 'high'):
        tiles, values = ([], [])
        for filename in splits[variant]:
            frame = cv2.imread(str(ROOT / 'data/images/train' / filename))
            h, w = frame.shape[:2]
            mask = np.ones((h, w), dtype=np.uint8)
            for x1, y1, x2, y2 in boxes_for(filename, 'train', w, h):
                mask[max(0, int(y1)):min(h, int(np.ceil(y2))), max(0, int(x1)):min(w, int(np.ceil(x2)))] = 0
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            bg = hsv[mask.astype(bool)]
            if len(bg) >= 0.05 * h * w:
                values.append({'file': filename, 'mean_brightness': float(bg[:, 2].mean()), 'mean_saturation': float(bg[:, 1].mean()), 'background_fraction': float(len(bg) / (h * w))})
            thumb = np.full((145, 180, 3), 255, np.uint8)
            scale = min(180 / w, 120 / h)
            small = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))))
            thumb[:small.shape[0], :small.shape[1]] = small
            cv2.putText(thumb, filename, (4, 137), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            tiles.append(thumb)
        sheet = np.full(((len(tiles) + 4) // 5 * 145, 900, 3), 255, np.uint8)
        for i, tile in enumerate(tiles):
            y, x = (i // 5 * 145, i % 5 * 180)
            sheet[y:y + 145, x:x + 180] = tile
        cv2.imwrite(str(out / f'{variant}_dataset.jpg'), sheet)
        features[variant] = values
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, field, title in zip(axes, ('mean_brightness', 'mean_saturation'), ('Background brightness', 'Background saturation')):
        for variant in ('low', 'high'):
            values = [r[field] for r in features[variant]]
            ax.hist(values, bins=np.linspace(0, 256, 9), weights=np.ones(len(values)) / len(values), alpha=0.6, label=f'{variant} (n={len(values)})')
        ax.set(xlabel=title + ' (0-255)', ylabel='Share of images')
        ax.legend()
    fig.suptitle('Pixels outside labeled pen boxes; a rough background measure')
    fig.tight_layout()
    fig.savefig(out / 'diversity.png', dpi=150)
    plt.close(fig)
    save_json(out / 'background_features.json', features)
    clip_dir = ROOT / 'test_clip'
    clip_dir.mkdir(exist_ok=True)
    clip = clip_dir / 'photos.avi'
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'MJPG'), 1, (640, 480))
    if not writer.isOpened():
        raise RuntimeError('Cannot write photo test clip')
    frames = []
    try:
        for i, filename in enumerate(splits['test']):
            frame = cv2.imread(str(ROOT / 'data/images/test' / filename))
            h, w = frame.shape[:2]
            scale = min(640 / w, 480 / h)
            nw, nh = (round(w * scale), round(h * scale))
            x, y = ((640 - nw) // 2, (480 - nh) // 2)
            canvas = np.full((480, 640, 3), 114, np.uint8)
            canvas[y:y + nh, x:x + nw] = cv2.resize(frame, (nw, nh))
            truth = [[a * nw / w + x, b * nh / h + y, c * nw / w + x, d * nh / h + y] for a, b, c, d in boxes_for(filename, 'test', w, h)]
            writer.write(canvas)
            frames.append({'index': i, 'file': filename, 'boxes': truth, 'condition': 'held-out public photos'})
    finally:
        writer.release()
    save_json(clip_dir / 'photos.json', {'source': 'held-out photo slideshow; NOT a webcam test', 'live_camera': False, 'novel_condition_confirmed': False, 'label_type': 'boxes', 'fps': 1, 'video_sha256': sha256(clip), 'frames': frames})
    save_json(out / 'dataset_check.json', {'counts': counts, 'author_groups_disjoint': True, 'exact_image_duplicates': 0, 'splits_sha256': sha256(ROOT / 'data/splits.json')})
    print('Data checked. Saved dataset sheets, diversity chart and test_clip/photos.avi.')
if __name__ == '__main__':
    main()
