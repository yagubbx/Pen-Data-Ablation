# Clip Comparison

Frames: 120. Confidence: 0.45.
Count unit: frames with/without any pen.

| Model | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| low | 0 | 0 | 75 | N/A | 0.0% |
| high | 1 | 0 | 74 | 100.0% | 1.3% |

See side_by_side.avi, frame images and per_frame.csv.
Green boxes are predictions. Blue boxes, when shown, are labels.
Presence scores do not measure box location or extra boxes in positive frames.

Low missed 75; high missed 74.
Low had 0 false positives; high had 0.
Check the saved frames to see which backgrounds or pen positions caused errors.
Collect more examples of those cases, plus empty scenes. Test on a new clip after retraining.
