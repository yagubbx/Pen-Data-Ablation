# Data and Experiment

| Split | Images | Use |
|---|---:|---|
| Low diversity | 16 | Train on mostly plain, light backgrounds |
| High diversity | 79 | Train on paper, desks, hands, dark and colored backgrounds |
| Shared validation | 16 | Check both models during training |
| Shared test | 29 | Compare final models only |

The low set is part of the high set. It was chosen by looking at the photos,
before model training. These are similar background conditions, not one camera session.
The lighting is visually similar, but camera settings are not known.
The high set adds 63 photos. It also changes pen types, angles and sizes.

Each photo is named `pen_1.jpg`, `pen_2.jpg`, and so on, with a matching `.txt` label.
Numbers stay the same across this project and the first detector project.
`data/splits.json` lists every split. `data/manifest.json` keeps source details.
Authors are kept in one split, and exact duplicate images are rejected.

Both models start from the same COCO pretrained YOLOv8n weights, not the previous
pen model. Both learn one custom class: `pen`.
Both use 30 epochs, seed 42, 416-pixel input, batch 8, AdamW, learning rate 0.0005,
and 10 frozen backbone layers. Both keep the last epoch.
Color augmentation is off for both; mild geometric augmentation is the same.
The full settings are in `reports/training_config.json`.

The high set has more images, so it also gets more training steps per epoch.
Low has 60 optimizer steps in total; high has 300.
This compares small/similar data with larger/varied data. It cannot measure the
effect of diversity alone. A size-matched study and more random seeds would be needed for that.

All comparisons use confidence 0.45, NMS IoU 0.45 and no temporal smoothing.
Image and photo-clip scores match boxes at IoU 0.5. Each label can match only one box.
Precision means the share of predicted boxes that are correct.
Recall means the share of labeled pens that were found.

`reports/diversity.png` shows brightness and color saturation outside labeled boxes.
Each histogram is normalized by its number of measured images.
Images with less than 5% background are skipped. Boxes may include background,
so this is only a rough measure; check the dataset sheets too.

`evaluate.py` also tests both models on low-training and high-only-training images.
These scores show familiar conditions. They are not held-out accuracy.

The fixed photo clip contains the 29 test images, one frame each, at 1 FPS.
It gives repeatable box-error counts. It is not a real webcam video.
The live recorder uses frame-level pen-present/absent labels.
These detect missing pens and false alarms on empty scenes, but not incorrect box locations
or extra boxes when a real pen is also visible. Label with P/N only when the scene is stable.
Press U before moving objects. Clip playback is fixed at 10 FPS; capture timestamps are saved.
The live FPS includes both models, capture and drawing. Offline inference times in reports
are local timings and are not live FPS measurements.

Background conditions in public photos are not fully known. For the live test,
review both contact sheets and choose clearly different setups, pens and lighting.
The `--novel` flag records your confirmation; it cannot verify the room automatically.

Further reading: [Data augmentation guide](https://www.v7labs.com/blog/data-augmentation-guide).
