# Dataset Source

These 124 photos and pen boxes come from [Open Images](https://storage.googleapis.com/openimages/web/download_v7.html).
The class is `Pen` (`/m/0k1tl`), from V5 validation/test box annotations.
This project makes a new split; these are not official benchmark results.

Photos use CC BY 2.0. Box labels use CC BY 4.0.
See the [license details](https://storage.googleapis.com/openimages/web/factsfigures_v7.html).
Each author's name, source URL, original image ID and file hash are in `manifest.json`.
The original photo bytes are unchanged. Boxes are in YOLO format.
Group boxes and boxes marked as drawings were excluded.

The files are named `pen_1.jpg` to `pen_124.jpg`; labels use the same numbers.
The manifest maps those names to the original sources.
The photo test clip and dataset sheets are resized views of these photos and keep their attribution.

The dataset is small and contains no separate pen-absent images.
Use empty scenes in your real webcam test to check false alarms.

YOLOv8 is provided by [Ultralytics](https://github.com/ultralytics/ultralytics).
The base checkpoint is COCO pretrained YOLOv8n. The two pen checkpoints are fine-tuned from it.
Ultralytics software and models use its licensing terms, including AGPL-3.0.
