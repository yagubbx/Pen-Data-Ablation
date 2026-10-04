# Start

Use Python 3.11, Git and a webcam. Run these commands in Windows PowerShell.
Replace `REPO_URL` with your repository link.

```powershell
git clone REPO_URL pen-data-ablation
cd pen-data-ablation
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python compare.py
```

Both trained models are included. The last command compares the fixed photo clip.
Open `reports/clip_photos/RESULTS.md` and `side_by_side.avi`.

## Your phone videos

The three files in `videos/` have reviewed time intervals in `videos/segments.json`.
To prepare their labels and compare both models, run:

```powershell
.venv\Scripts\python phone_tests.py
```

Read `PHONE_RESULTS.md`. Open the comparison videos in `reports/phone/`.
This is an offline phone-video test. It does not replace the live test below.

## Live webcam test

Look at `reports/low_dataset.jpg` and `reports/high_dataset.jpg`.
Choose a new room, light and background. Change the description below to match it.

```powershell
.venv\Scripts\python live.py --condition "Blue cloth, side lamp, my room" --novel
```

- Put a pen in view. Press **P**, then keep it there for 15 seconds.
- Press **U** before moving the pen out of view.
- Remove the pen. Press **N**, then wait 15 seconds.
- Press **Q**. Wait for the saved clip comparison to finish.

Open the saved session in `captures/`. Its `comparison/` folder has
FP/FN counts, screenshots and a comparison video.
Repeat with two more new setups, such as dim light and a busy desk.
Write what you saw in `LIVE_RESULTS.md`. Camera issue? Add `--source 1`.

## Optional commands

```powershell
.venv\Scripts\python evaluate.py
.venv\Scripts\python prepare.py
.venv\Scripts\python train.py --overwrite
.venv\Scripts\python -m unittest -v test_metrics
```

`evaluate.py` checks shared test images. `prepare.py` rebuilds the charts and photo clip.
`train.py` trains both models again. It replaces the weights; run both evaluations again after training.
Update `RESULTS.md` if the results change.
