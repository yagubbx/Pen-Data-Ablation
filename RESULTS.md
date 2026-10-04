# Results

Both models were trained for 30 epochs from the same YOLOv8n base weights.
These are measured offline results. The completed webcam test is in [LIVE_RESULTS.md](LIVE_RESULTS.md).

The three new phone recordings are measured in [PHONE_RESULTS.md](PHONE_RESULTS.md).
See [PHONE_REVIEW.md](PHONE_REVIEW.md) for correct detections and wrong boxes.

## Shared test images

29 held-out photos. Confidence 0.45; correct box = IoU at least 0.5.

| Model | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| low | 0 | 0 | 49 | N/A | 0.0% |
| high | 26 | 4 | 23 | 86.7% | 53.1% |

TP = correct box. FP = wrong or extra box. FN = missed labeled pen.
See `reports/evaluation/per_image.csv` and `reports/evaluation/examples/`.
Green boxes are predictions; blue boxes are labels.

The low model made no boxes above the threshold on these test images.
Its zero false positives do not mean good performance: it missed every labeled pen.
It also had low recall on its own training set, so limited learning and the
confidence threshold matter here, as well as the data conditions.

## What the images show

- `pen_100.jpg`: high found the close pen on a light surface; low missed it.
- `pen_103.jpg`: high found 2 of 6 pens on a dark background; low missed all 6.
- `pen_109.jpg`: both missed the pen on the keyboard.
- `pen_119.jpg`: high found the pen but also drew an extra box on part of it.
- `pen_124.jpg`: both missed the pen pointing toward the camera.

## Same fixed clip

29 frames from the same held-out photos, resized and saved as a video.
Both models read every frame. This is a photo slideshow, not a webcam test.
Resizing and video compression can change predictions.

| Model | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| low | 0 | 0 | 49 | N/A | 0.0% |
| high | 23 | 3 | 26 | 88.5% | 46.9% |

See `reports/clip_photos/side_by_side.avi`, frame images and `per_frame.csv`.

## Familiar training conditions

Low-set photos (both models saw these during training):

| Model | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| low | 3 | 0 | 29 | 100.0% | 9.4% |
| high | 27 | 0 | 5 | 100.0% | 84.4% |

High-only photos (only the high model saw these during training):

| Model | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| low | 2 | 0 | 78 | 100.0% | 2.5% |
| high | 71 | 0 | 9 | 100.0% | 88.8% |

These two checks are training-set scores, not held-out performance.

## Conclusion

On the shared test images, the low model missed 49 pens and the high
model missed 23. False positives were 0 and 4.
The larger, varied set helped the model find more pens in this run.
Collect photos in different rooms, with bright and dim light, different angles,
and different pen sizes. Add empty scenes with similar objects to check false alarms.
Keep a new room or camera session for testing. Use the live errors to choose what to collect next.

This experiment changes both image count and diversity. The larger set also has
more training steps. One seed, a small test set and possible label errors limit the result.
There were 60 optimizer steps for low and 300 for high (same 30 epochs).
It does not prove that diversity alone caused the difference, or that either model works everywhere.

The background chart is in `reports/diversity.png`.
Training settings, curves and checkpoint hashes are in `reports/`.
See [LIVE_RESULTS.md](LIVE_RESULTS.md) for the completed live test.
