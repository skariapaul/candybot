# Real "Hey Zen" recordings

40 real recordings of "Hey Zen" (34 train / 6 test), one speaker, varied
volume/pace/mic distance, captured on the dev laptop's Poly Blackwire 3325
headset on 2026-07-22.

## Why these exist

The first `hey_zen.onnx` model, trained entirely on synthetic Piper TTS
clips, scored well on its own synthetic validation set (51.6% recall) but
produced an almost perfectly flat, unresponsive score (~0.0005-0.0007) on
real microphone audio -- including during clear, deliberate "hey zen"
utterances. Retraining with different `max_negative_weight`/
`target_false_positives_per_hour` settings and adding runtime gain
normalization (AGC) didn't fix it. That points to a synthetic-to-real
generalization gap: the model learned Piper's TTS voice distribution, not
real human speech through a real mic. These recordings are meant to close
that gap by giving the model *some* real-audio exposure alongside the much
larger synthetic set.

Note: ~7 of the 40 clips clip at full-scale (loud takes) -- acceptable, not
re-recorded, since some clipping is normal in real-world audio anyway.

## How to fold these into a training run

The augment step just globs `*.wav` in the trainer's `positive_train`/
`positive_test` output directories (regardless of whether files are
synthetic or real), so no code changes are needed -- just copy these files
in before re-running `augment`, on the MI300X training host:

```bash
cp training/wakeword/real_samples/hey_zen/positive_train/*.wav \
   training/wakeword/trainer/output/hey_zen/positive_train/
cp training/wakeword/real_samples/hey_zen/positive_test/*.wav \
   training/wakeword/trainer/output/hey_zen/positive_test/

cd training/wakeword/trainer
python train_wakeword.py --config configs/hey_zen.yaml --from augment
```

`step_augment` already runs with `--overwrite`, so features get
regenerated including these files without needing to redo the (slow)
`generate` step at all.
