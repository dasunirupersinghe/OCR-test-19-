# Pilot reference model

`prototypes.npz` and `prototypes.json` form one checked model. Both are required.
The archive contains 30 whole-glyph classes with 46 features, global feature
standardization and regularized per-class covariance matrices. No independent
upper/lower modifier models, lexicon, bigram corpus or domain corpus were fitted.

Calibration used 20 images per class (600 total), reserving five per class
(150 total). The saved model was loaded and matched 119 held-out labels: 79.3%
isolated-character accuracy on this random split. No writer identifiers are
available. This is a partial-alphabet research pilot, not a validated Sinhala
word recognizer.

The source is the upstream `training_set` at:
https://github.com/asithishantha/Sinhala-Handwriting-Character-Recognition-based-on-Feature-Extraction/tree/d2c87bb1eed2d6177ffda4ad07f79e38cf1d993d

The active ordered `letters` array in `myFeatureScript.m` was joined to indexed
Unicode values in `test.m`. Thirty Sinhala labels were verified; the ASCII
classes `o` (literal `O`), `num3`, `num4` and `num5` were excluded. No dataset
license was specified upstream. Raw training and test images are not bundled.
`dataset_provenance.json` records the mapping evidence and source checksums.

`calibration_report.json` records all calibration/holdout file hashes, label
predictions and successful model round-trip verification. The model is 209,290
bytes; SHA-256:

```text
e10e4243af494b46ff8a614702eb8dff9e9dcff2d79e513817ac74681b853877
```

`sample_word_results.json` records three independent public sample tests.
Manual readings were `ගම`, `රට` and `මට කන්න ඕනේ`; actual outputs were `ඉ`,
`ළග` and `ණ`. All failed. Manual references are not authoritative dataset
labels, and these few cases do not define a population word-error rate.
No sample hash occurs in the calibration or held-out glyph split.

The first case merged a two-character word into one glyph proposal. The second
produced two proposals but classified them incorrectly. The photographed line
was merged into one word/glyph proposal. Training and query crop preparation
also differ. Near-one relative class scores and `needs_review=false` did not
indicate correct recognition. The requested mean-max confidence rule does not
provide an out-of-distribution rejection mechanism.

`recognition_result.json` is a separate run of `recognize_image.py` against the
saved model and the first public sample. It confirms the same incorrect output.
Paths in reports identify files used in the cloud run; pinned source links and
checksums support reproduction elsewhere. Chat-uploaded images were not tested
because they were not available as local input files.
