# Offline Sinhala OCR research modules

This source archive contains Files 1–5 of the project:

- `config.py`: validated B0–B4 ablation presets and Sinhala Unicode tables.
- `preprocessing.py`: quality checks, deskewing, enhancement and binary masks.
- `segmentation.py`: word localization, normalization, spatial zones and cut hypotheses.
- `features.py`: micro-window slope and topology descriptors.
- `decoder.py`: prototype calibration, classical quantum-inspired scoring, Sinhala assembly and an OOV-safe lexicon gate.

These are Python library modules. A graphical interface, trained model, labelled
dataset and lexicon are not included. Recognition needs labelled calibration
examples prepared consistently with query glyphs. The quantum-inspired scorer
runs on a normal CPU; its normalized class scores do not establish OCR accuracy
or calibrated probabilities of correctness. Validate on held-out Sinhala data.

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
held-out evaluation data separate. This complete library example accepts your
actual data and does not supply a fabricated pretrained model:

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
