# Clip Comparison

Frames: 29. Confidence: 0.45.
Count unit: objects at IoU >= 0.5.

| Model | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| low | 0 | 0 | 49 | N/A | 0.0% |
| high | 23 | 3 | 26 | 88.5% | 46.9% |

See side_by_side.avi, frame images and per_frame.csv.
Green boxes are predictions. Blue boxes, when shown, are labels.
This photo clip is an offline test. It does not prove live webcam performance.

Low missed 49; high missed 26.
Low had 0 false positives; high had 3.
Check the saved frames to see which backgrounds or pen positions caused errors.
Collect more examples of those cases, plus empty scenes. Test on a new clip after retraining.
