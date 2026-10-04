# Phone Video Review

These notes describe the included run at confidence 0.45.
They are visual checks, not full bounding-box annotations for every frame.
If you retrain the models, review the new predictions again.

| Recording | What happened? |
|---|---|
| Plain background | High found the closer pen in several frames. Low made no boxes. |
| Dim, cluttered background | High's only presence hit was a wrong box on the glasses case. The pen was missed in that frame. |
| Normal light, cluttered background | High sometimes found the pen, but some boxes also covered nearby objects. Low made no boxes. |

Examples:

- [Plain, 22.20 seconds](reports/phone/test_1_plain/frame_71.jpg): high puts a box around the pen.
- [Dim, 22.81 seconds](reports/phone/test_2_dim_clutter/frame_64.jpg): high labels the glasses case as a pen.
- [Normal light, 12.21 seconds](reports/phone/test_3_bright_clutter/frame_41.jpg): high finds the pen.
- [Normal light, 24.22 seconds](reports/phone/test_3_bright_clutter/frame_71.jpg): high's box is too wide and includes nearby objects.

Both models made zero boxes in the selected pen-absent frames. This does not mean
zero object-detection errors: the dim example has a wrong box while a real pen is present.
The presence table cannot count that as an FP and FN. Full box labels would be needed
for object-level precision and recall on these videos.

The high model was more useful on these views, but it still missed many pens.
The low model's zero detections also show weak learning at this threshold;
this is not evidence that simple backgrounds alone caused the failure.
The training sets differ in image count, diversity and total optimizer steps.

Next, collect labeled thin light pens on dark surfaces, close and far views,
dim scenes, and glasses cases without pens. Keep a new recording for the next test.

All three recordings were evaluated after recording, not live through a webcam.
A separate live webcam session is documented in [LIVE_RESULTS.md](LIVE_RESULTS.md).
