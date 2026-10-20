# Offline Sinhala OCR research modules

This source archive contains Files 1–5 of the project:

- `config.py`: validated B0–B4 ablation presets and Sinhala Unicode tables.
- `preprocessing.py`: quality checks, deskewing, enhancement and binary masks.
- `segmentation.py`: word localization, normalization, spatial zones and cut hypotheses.
- `features.py`: micro-window slope and topology descriptors.
- `decoder.py`: prototype calibration, classical quantum-inspired scoring, Sinhala assembly and an OOV-safe lexicon gate.

The repository also includes CPU calibration and recognition commands and a
**30-class pilot reference model** in `artifacts/`. This is a partial Sinhala
inventory, and the independent word tests failed. It is not ready for reliable
word or document recognition. A graphical interface, raw dataset and lexicon
are not included. The quantum-inspired scorer runs on a normal CPU; its
normalized class scores are not calibrated probabilities of correctness.

## Install

The source uses Python 3.10+ syntax. Use Python 3.11 or newer with the exact
dependency versions in this archive because these dependency releases require
Python 3.11+. This archive was checked with Python 3.12.14. No GPU is required.

Extract this ZIP, open a terminal, and enter its `OCR-test-19-` directory.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

If PowerShell prevents activation, use `.\.venv\Scripts\python.exe` in place
of `python` in the remaining commands; activation is optional.

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The included tests exercise decoder contracts using synthetic inputs. Passing
them confirms those contracts, not recognition accuracy on real handwriting.

## Calibrated pilot and actual results

Cloud calibration completed using 600 real isolated handwriting images across
30 Unicode classes. A separate random holdout of 150 images produced **119
correct classifications (79.3%)**. Writer identities are unavailable, so this
is not a writer-independent evaluation or a word-recognition accuracy estimate.

The saved model was loaded and run on independent public word images:

| Manual visual reference | Actual decoder output |
| --- | --- |
| ගම | ඉ |
| රට | ළග |
| මට කන්න ඕනේ | ණ |

These references are manual readings, not author-provided ground-truth labels.
The samples were not used for calibration or the isolated-character holdout.
The outputs received high relative class scores despite being incorrect;
`needs_review=false` is therefore not evidence of correctness. The tests expose
both segmentation and classification failures. Chat-uploaded images were not
available as local files for this run; these are separate public test images.

Keep [prototypes.npz](artifacts/prototypes.npz) and its checked
[JSON sidecar](artifacts/prototypes.json) together. The archive is 209,290 bytes,
well below the 2,000,000-byte limit. The
[calibration report](artifacts/calibration_report.json) records the full split,
labels, predictions, hashes and successful save/load validation. The
[word-test report](artifacts/sample_word_results.json) contains pinned source
links, image hashes and actual outputs. The
[CLI recognition report](artifacts/recognition_result.json) reproduces the first
word failure. See [artifact notes](artifacts/README.md) for limitations.

Run the saved model on an image:

```bash
python recognize_image.py --image my_word.png --model artifacts/prototypes.npz --output recognition_result.json
```

Reproduce dataset preparation and calibration in a separate output directory:

```bash
python prepare_reference_data.py --destination data/reference
python calibrate_prototypes.py --data-dir data/reference --label-map data/reference/labels.json --output artifacts/recalibrated/prototypes.npz --mode B4 --holdout-per-class 5 --max-samples-per-class 20 --seed 42 --source-url https://github.com/asithishantha/Sinhala-Handwriting-Character-Recognition-based-on-Feature-Extraction/tree/d2c87bb1eed2d6177ffda4ad07f79e38cf1d993d --source-license "Not specified in source repository; raw images are not redistributed"
```

Preparation verifies the pinned upstream revision, the MATLAB folder-to-Unicode
mapping and all 750 selected images. It excludes four ASCII classes rather than
guessing Sinhala labels. Raw images stay in ignored local data directories.
The upstream repository does not specify a dataset license; this project does
not redistribute its images. The pilot contains whole-glyph prototypes, with
no separately calibrated upper/lower modifier models and no language corpus.
Improving recognition requires broader labelled coverage, consistent training
and query crops, and a separate word-level evaluation set.

## Strict 46-dimensional calibration and decoding

`DevelopmentSetCalibrationManager` implements the explicit Master Specification
profile: 16 slope angles, 12 topology descriptors, and nine hole-centroid pairs.
It does not pad or truncate the earlier 36/51-value schemas. Use its
`feature_pipeline` for queries and recalibrate for this schema.

Prototype means and distances use development-fitted global standardization.
`book.raw_means` exposes means in the original feature space. The strict manager
uses unshrunk sample covariance plus `cov_reg_eps * I`; two samples suffice
because regularization handles rank deficiency. A single sample or an unusable
covariance has an explicit standardized Euclidean fallback. B0 always uses
standardized Euclidean distance. Amplitudes and probabilities are calculated
stably in probability-enabled presets; original baseline flags are respected.

The 46-value feature profile explicitly enables both feature branches, with its
own schema/configuration identity. The original whole-pipeline B0–B4 feature
ablations remain available through the generic `FeaturePipeline` and
`PrototypeCalibrationManager`; do not report a feature override as an unchanged
ablation preset.

Development folders need an explicit folder-to-Sinhala-Unicode label map.
Numerical dataset IDs are never guessed. Keep development/calibration data and
held-out evaluation data separate. This library example accepts your own
labelled development data:

```python
from pathlib import Path
from typing import Mapping

from decoder import DecoderPipeline, DevelopmentSetCalibrationManager
from preprocessing import PreprocessingPipeline
from segmentation import SegmentationPipeline

manager = DevelopmentSetCalibrationManager(cov_reg_eps=1e-4)

def calibrate(development_root: Path, label_map: Mapping[str, str]):
    book = manager.calibrate_from_folders(development_root, label_map)
    book.save("prototypes.npz")
    return book

def decode(image_path: Path, book, lexicon=None):
    mask, metadata = PreprocessingPipeline(manager.config).run(image_path)
    if metadata["status"] != "ok":
        raise ValueError("The image contains no usable handwriting")
    segmentation = SegmentationPipeline(manager.config).run(mask)
    batch = manager.feature_pipeline.run(segmentation)
    return DecoderPipeline(book, lexicon=lexicon, config=manager.config).run(batch)
```

Inspect returned word statuses and diagnostics before accepting output. Input
binary masks use ink=1 (or 255), background=0. OpenCV arrays default to BGR/BGRA;
pass `color_order="rgb"` when preprocessing RGB/RGBA arrays.

## Core and modifier models in one NPZ

`DevelopmentFolderSpec` declares full isolated character images and labels for
one target view. For core/upper/lower views the strict manager extracts the
corresponding normalized HPP region automatically. Each view needs its own
explicit labels: a base consonant, an actual modifier, or a learned empty-string
null class for nonempty upper/lower ink that is not a modifier. Empty zones do
not need a prototype. Already-sliced or custom-prepared inputs require the
`feature_builder` callback to match query preparation.

```python
from decoder import DevelopmentFolderSpec, PrototypeBundle

def calibrate_zones(core_root, core_labels, upper_root, upper_labels,
                    lower_root, lower_labels):
    bundle = manager.calibrate_bundle_from_folders({
        "zone:core": DevelopmentFolderSpec(core_root, core_labels),
        "zone:upper": DevelopmentFolderSpec(upper_root, upper_labels),
        "zone:lower": DevelopmentFolderSpec(lower_root, lower_labels),
    })
    bundle.save("zone_prototypes.npz")
    return PrototypeBundle.load("zone_prototypes.npz")
```

Pass this mapping directly to `DecoderPipeline`. It contains per-view class
means, full covariance matrices, scales, labels and exact schema identities.
`bundle.modifier_signatures["zone:upper"][label]` exposes a read-only raw R46
signature. Keep each NPZ and its checked JSON sidecar together.

Every saved NPZ must be **strictly smaller than 2,000,000 bytes**, checked before
writing. Storage is lossless float64. Expanded arrays have a separate 256 MiB
loading budget and their headers are validated before NumPy allocation. Some
large full-covariance inventories cannot fit under 2 MB; saving fails explicitly
instead of silently quantizing or discarding classes. Factorized core/modifier
models are usually smaller than hundreds of complete consonant-vowel classes.

Native modifier ownership is recomputed from horizontal box overlap; ambiguous
split crops are rejected while intact hypotheses remain available. Independently
extracted modifiers can be supplied with
`DecoderPipeline.run(batch, modifiers=observations)`, where observations are
`ModifierCandidate` objects carrying zone features and normalized word boxes.
`HorizontalZoneBinder` assigns each once to the unique greatest positive x
overlap. Touching edges have zero overlap; ties and unbound marks are rejected.
External observations replace native upper/lower crops to prevent double scoring.

Sinhala assembly reorders visual kombuwa into logical order, combines composite
signs, preserves explicit joiners, and NFC-normalizes output. The B4 lexicon gate
preserves raw output when the specified mean-max class confidence is at least
0.85. Otherwise it considers at most three scored model paths within one NFC
Unicode codepoint edit; unscored dictionary neighbours never receive invented
probabilities. `selected_label_confidence` is reported separately when Unicode
filtering selects a lower-ranked label. These scores do not establish recognition
accuracy. With no lexicon, the raw model output is retained.

## Two-pass context-aware word and line decoding

`DecoderPipeline.decode_word_or_line(batch, doc_type=...)` and `run` first
recognize the complete line, then perform two context passes. The standalone
`decode_word_or_line(batch, models, ...)` helper provides the same operation.

Pass 1 locks every nonempty word with original confidence at least 0.85,
including proper nouns absent from the lexicon. With B4 gating disabled, words
are retained as policy anchors. Pass 2 changes only ambiguous B4 words. It uses
the immediately adjacent Pass 1 anchors; a corrected word does not become a new
anchor during that call. Rejected or empty word slots preserve their position
and never permit a context link across the gap.

For each of the first three original model paths that occurs in the lexicon
and is within one NFC codepoint edit, the score is exactly:

```text
d_edit(raw, candidate)
  - eta_lex * log(P_candidate + 1e-12)
  - beta_left * log(P(candidate | left_anchor) + 1e-6)
  - beta_right * log(P(right_anchor | candidate) + 1e-6)
  - beta_doc * log(P(candidate | doc_type) + 1e-6)
```

The original path probability is preserved; the three candidates are not
renormalized. `logaddexp` evaluates the model term without word-probability
underflow. Missing neighbors, context sources or document type disable the
corresponding weights. An observed zero probability uses the stated epsilon.
An isolated word therefore has exactly zero left/right contributions.

Supply offline context tables from development data. Bigram keys are
`(previous_word, next_word)` and domain rows are indexed by document type:

```python
from decoder import ContextProbabilityModel, DecoderPipeline

def decode_with_context(feature_batch, prototypes, lexicon,
                        bigram_counts, domain_counts, doc_type=None):
    context = ContextProbabilityModel.from_counts(
        bigrams=bigram_counts, domains=domain_counts, split="development")
    pipeline = DecoderPipeline(prototypes, lexicon=lexicon,
                               config=manager.config, context_model=context)
    return pipeline.decode_word_or_line(feature_batch, doc_type=doc_type)
```

You can also pass already-normalized tables directly to
`ContextProbabilityModel(bigrams=..., domains=...)`. Rows are validated and
copied into immutable tables. No external model or corpus is downloaded.
Context tables are separate from the size-limited prototype NPZ artifact.
For precomputed word paths, `TwoPassContextDecoder.decode_word_or_line` accepts
one sequence of `WordPath` objects or one sequence per word of a line.

Configure `decoder.eta_lex`, `beta_left`, `beta_right`, and `beta_doc` through
`get_config(..., overrides=...)`. The beta defaults are 0.10; `eta_lex=None`
reuses the existing `eta` value. Zero disables the corresponding term.

Each returned word exposes `needs_review` and `is_context_anchor`, and the result
metadata lists `review_word_ids`. Missing confidence, unavailable model scores,
or no eligible lexicon path preserves the raw text and requests human review.
The word's confidence stays the original mean-max recognition confidence after
correction. Candidate scores, score components, anchor neighbors and effective
weights are available in `word.metadata["context_decision"]` for inspection.
