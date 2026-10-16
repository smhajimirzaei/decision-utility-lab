# Contributing

Start with the CPU suite and one recorded audit; neither needs model weights:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -S scripts/summarize_trajectory_study.py --input docs/data/trajectory_study.json --output runs/check
```

A bug report should include the command, Python version, input schema, and a small reproducer. For numerical changes, explain which weighting or intervention contract changes and compare the affected recorded results.

Preserve matched repeat identities, logical candidate weights, episode/seed splits, and the distinction between offline controls and deployment methods. Do not replace negative results with a selected winner, count branches as independent episodes, or put outcomes/actual futures into deployment features.

Add a test for a new numerical contract or a regression that could change a decision. Presentation edits can be checked by rebuilding their output. Use a fresh directory under `runs/`; generated runs are ignored.

Keep model downloads, raw archives, credentials, personal paths, and cloud orchestration out of commits. Edit `scripts/demo_template.html` and rebuild the recorded demo rather than hand-editing its embedded data.
