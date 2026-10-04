# Phone Video Results

Three phone recordings were tested offline. See LIVE_RESULTS.md for the separate live test.
Confidence: 0.45. Input: 416. No smoothing. Both models read the same saved frames.
Five frames per second were selected in visually labeled intervals.
Unclear transitions were excluded before looking at predictions.

| Video | Model | Positive / negative frames | TP | FP | FN | TN | Presence precision | Presence recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| test_1_plain | low | 105 / 35 | 0 | 0 | 105 | 35 | N/A | 0.0% |
| test_1_plain | high | 105 / 35 | 49 | 0 | 56 | 35 | 100.0% | 46.7% |
| test_2_dim_clutter | low | 75 / 45 | 0 | 0 | 75 | 45 | N/A | 0.0% |
| test_2_dim_clutter | high | 75 / 45 | 1 | 0 | 74 | 45 | 100.0% | 1.3% |
| test_3_bright_clutter | low | 95 / 40 | 0 | 0 | 95 | 40 | N/A | 0.0% |
| test_3_bright_clutter | high | 95 / 40 | 28 | 0 | 67 | 40 | 100.0% | 29.5% |

FP: a box when no pen is visible. FN: no box when a pen is visible.
These are frame-level presence counts, not box-location scores.
A wrong box in a pen-present frame can still count as a presence hit.
A 100% presence precision does not mean all boxes are correct.
See [PHONE_REVIEW.md](PHONE_REVIEW.md) for visual checks of the included run.
Frames from one video are related, so these are not independent examples.

## Files

Originals: `videos/`. Reviewed intervals: `videos/segments.json`.
Prepared clips and per-frame labels: `test_clip/phone/`.
Reports, screenshots and side-by-side videos: `reports/phone/`.

The original phone files are unchanged. The prepared clips use MJPEG at 5 FPS.
The supplied phone files are 478 x 850 pixels. Source timestamps are saved in the labels.
These new recordings were not used to train either model. Dim and bright clips share
a surface, objects and pen, so they are a small lighting comparison, not separate rooms.

## Next data to collect

Use the missed views to collect more thin light pens on dark surfaces, small pens,
and clutter under different lighting. Include empty scenes. Label pen boxes, train
again, and test on a new recording. Keep these videos out of training if they remain the test set.

For the live requirement, run `live.py` with a webcam as described in START.md.
