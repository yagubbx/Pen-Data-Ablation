# Live Webcam Results

Completed: one live webcam session in the user's room, with normal light.
The user confirmed a new setup and labeled frames with N, U and P.
Both models ran on the same camera frames. Confidence: 0.45; input: 416; no smoothing.

The saved clip was then replayed through both models for fixed-frame counts.
All 308 frames, labels, CSV totals and clip hashes were checked before media removal.

| Model | TP | FP | FN | TN | Presence precision | Presence recall |
|---|---:|---:|---:|---:|---:|---:|
| Low | 0 | 0 | 158 | 150 | N/A | 0.0% |
| High | 78 | 0 | 80 | 150 | 100.0% | 49.4% |

There were 158 pen-present and 150 pen-absent frames.
High produced boxes in more pen-present frames. Low produced no boxes.
Neither model produced boxes in the selected pen-absent frames.
These are presence scores, not box-location scores: a wrong box in a positive frame
can count as a presence hit. The boxes were not fully annotated or visually verified
for this live session. A 100% presence precision does not mean perfect detection.

The webcam videos and all extracted live-session images were deleted at the user's
request to avoid including their face. Only numeric results are included.
The removed videos cannot be replayed from this repository. The JSON retains their
hashes as records of the checked run; a hash cannot replace visual evidence.

Results: [metrics.json](reports/live/metrics.json) and [per_frame.csv](reports/live/per_frame.csv).
The `captures/` folder stays ignored by Git. Do not force-add private camera sessions.
Computer and webcam model details were not recorded. Offline inference timings in
metrics.json are not measurements of the full live-loop FPS.

The three phone recordings are separate offline tests; see [PHONE_RESULTS.md](PHONE_RESULTS.md).
Their visual review is in [PHONE_REVIEW.md](PHONE_REVIEW.md).

Recommendation: collect more varied pen sizes, angles, dark backgrounds and dim light,
plus empty scenes. Use a new camera session for the next test. The current comparison
also changes training-set size and optimizer steps, so it does not isolate diversity alone.
