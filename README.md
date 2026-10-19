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

## Calibrate and decode

Calibration accepts explicitly labelled `FeatureResult` objects. Build each
whole-glyph feature with the same `FeaturePipeline`, preset and image-preparation
policy as the candidate crops used for queries. Labels are actual Sinhala
Unicode strings, for example `"ක"`; numerical dataset IDs need an explicit
label map. Keep calibration/development and held-out evaluation data separate.

```python
from collections.abc import Iterable

from config import get_config
from decoder import CalibrationSample, DecoderPipeline, PrototypeCalibrationManager
from features import FeatureBatchResult, FeaturePipeline, FeatureResult
from segmentation import SegmentationPipeline

config = get_config("B4")
features = FeaturePipeline(config)

def calibrate(labelled_features: Iterable[tuple[str, FeatureResult]]):
    # Obtain these FeatureResults from your labelled, consistently prepared
    # isolated glyph masks, e.g. features.extract(mask, view="whole_glyph").
    samples = [CalibrationSample(label, feature, split="development")
               for label, feature in labelled_features]
    return PrototypeCalibrationManager(config).fit(samples)

def decode(binary_ink_mask, book, lexicon=None):
    # 2-D mask: foreground ink=1 (or 255), background=0.
    segmentation = SegmentationPipeline(config).run(binary_ink_mask)
    batch = features.run(segmentation)
    assert isinstance(batch, FeatureBatchResult)
    result = DecoderPipeline(book, lexicon=lexicon, config=config).run(batch)
    # Inspect each word's status/diagnostics before accepting its output.
    return result

# Supply your own labelled FeatureResults, input mask and optional lexicon:
# book = calibrate(labelled_features)
# book.save("prototypes.npz")  # also writes checked JSON metadata
# result = decode(binary_ink_mask, book, lexicon=known_sinhala_words)
# print(result.text)
```

For image files, `PreprocessingPipeline(config).run("word.png")` returns a
binary ink mask and metadata. Check the metadata status before segmentation.
Preprocessing outputs ink=1/background=0. OpenCV arrays default to BGR/BGRA;
pass `color_order="rgb"` when preprocessing RGB/RGBA arrays.

Whole-glyph decoding is the simplest initial model path. Multi-zone decoding
requires separately calibrated core/upper/lower prototype books with matching
feature schemas and view labels. A zone containing ink needs its matching book.
Prototype files preserve their schema and image-view identity and are rejected
on incompatible use. Keep the NPZ and accompanying JSON file together.

The B4 lexicon gate can select only a scored model alternative within one NFC
Unicode codepoint edit of the raw output, and it preserves high-confidence OOV
output. It does not invent a score for a dictionary neighbour. With no lexicon,
raw model output is retained. Read module docstrings for algorithm assumptions,
limits, diagnostics and configuration details.
