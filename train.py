"""Train both variants with one configuration and the same starting weights."""
import argparse
import shutil
import time
from common import ROOT, dataset, save_json, set_threads, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    if a.epochs < 1 or a.threads < 1:
        p.error('epochs and threads must be positive')
    if any((ROOT / f'weights/{n}.pt').exists() for n in ('low', 'high')) and not a.overwrite:
        p.error('Weights already exist. Use --overwrite to train again.')
    import yaml
    import torch
    import ultralytics
    from ultralytics import YOLO
    splits, counts = dataset()
    set_threads(a.threads)
    base = ROOT / 'weights/base.pt'
    if not base.is_file():
        raise FileNotFoundError('Missing weights/base.pt')
    config = dict(epochs=a.epochs, imgsz=416, batch=8, device=a.device, workers=0,
                  seed=42, deterministic=True, patience=0, optimizer='AdamW', lr0=0.0005,
                  nbs=8, freeze=10, warmup_epochs=0, warmup_bias_lr=0, degrees=10,
                  translate=0.05, scale=0.2, hsv_h=0, hsv_s=0, hsv_v=0, fliplr=0.5,
                  mosaic=0, close_mosaic=0, cache=False, conf=0.1, plots=True)
    save_json(ROOT / 'reports/training_config.json', {**config, 'threads': a.threads,
              'base_sha256': sha256(base), 'torch': torch.__version__,
              'ultralytics': ultralytics.__version__, 'checkpoint': 'last epoch'})
    for name in ('low', 'high'):
        folder = ROOT / 'runs' / name
        folder.mkdir(parents=True, exist_ok=True)
        for split, names in [('train', splits[name]), ('val', splits['val'])]:
            (folder / f'{split}.txt').write_text('\n'.join(
                (ROOT / 'data/images' / split / f).as_posix() for f in names)+'\n', encoding='utf-8')
        spec = {'path': ROOT.as_posix(), 'train': (folder/'train.txt').as_posix(),
                'val': (folder/'val.txt').as_posix(), 'names': {0: 'pen'}}
        data = folder / 'data.yaml'
        data.write_text(yaml.safe_dump(spec), encoding='utf-8')
        start = time.perf_counter()
        model = YOLO(str(base))
        model.train(data=str(data), project=str(folder), name='fit', exist_ok=True, **config)
        # A fixed last epoch prevents separate best-epoch choices from changing the comparison.
        output = ROOT / f'weights/{name}.pt'
        checkpoint = torch.load(model.trainer.last, map_location='cpu', weights_only=False)
        # Keep the portable experiment settings, without machine-specific paths.
        args = checkpoint.get('train_args', {})
        args.update(data=f'runs/{name}/data.yaml', model='weights/base.pt',
                    project='runs', name=name, save_dir=f'runs/{name}/fit')
        checkpoint['train_args'] = args
        for key in ('model', 'ema'):
            net = checkpoint.get(key)
            if net is not None and hasattr(net, 'args'):
                net.args = dict(args)
        torch.save(checkpoint, output)
        report = ROOT / 'reports/training' / name
        report.mkdir(parents=True, exist_ok=True)
        for filename in ('results.csv', 'results.png'):
            src = model.trainer.save_dir / filename
            if src.exists():
                shutil.copy2(src, report / filename)
        save_json(report / 'record.json', {'variant': name, 'train_images': counts[name],
                  'val_images': counts['val'], 'epochs': a.epochs,
                  'elapsed_seconds': round(time.perf_counter()-start, 2),
                  'weights_sha256': sha256(output), 'base_sha256': sha256(base),
                  'config_sha256': sha256(ROOT/'reports/training_config.json'),
                  'splits_sha256': sha256(ROOT/'data/splits.json')})
        print(f'Saved weights/{name}.pt', flush=True)


if __name__ == '__main__':
    main()
