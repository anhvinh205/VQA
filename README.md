# Visual Question Answering (Yes/No)

An image-question demo built with FastAPI and PyTorch. Upload an image and ask
whether a supported object is present. The default backend uses a pretrained
COCO object detector and automatically uses CUDA when PyTorch detects a
compatible GPU; otherwise it runs on CPU. The original CNN + BiLSTM classifier
is kept as a comparison baseline.

## What This Project Can Do

The default backend recognizes objects from the COCO dataset. It can answer a
question such as `Is there a dog in this image?` by detecting dogs in the image
and comparing the detector score with a threshold. No GPU is required; an
available CUDA GPU can accelerate inference when the installed PyTorch build
supports it.

It is **not a general-purpose visual question answering model**. It does not
understand color, actions, counts, or relationships between objects. For
example, `Is the person wearing a tie?` is not supported: the detector can find
people and ties independently but cannot determine whether the person is
wearing the tie. It also does not interpret negation: phrase questions as
affirmative presence checks such as `Is there a dog?`, not `Is there no dog?`.
Ask about one object category at a time.

The detector recognizes these COCO categories:

`person`, `bicycle`, `car`, `motorcycle`, `airplane`, `bus`, `train`, `truck`,
`boat`, `traffic light`, `fire hydrant`, `stop sign`, `parking meter`, `bench`,
`bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`,
`giraffe`, `backpack`, `umbrella`, `handbag`, `tie`, `suitcase`, `frisbee`,
`skis`, `snowboard`, `sports ball`, `kite`, `baseball bat`, `baseball glove`,
`skateboard`, `surfboard`, `tennis racket`, `bottle`, `wine glass`, `cup`,
`fork`, `knife`, `spoon`, `bowl`, `banana`, `apple`, `sandwich`, `orange`,
`broccoli`, `carrot`, `hot dog`, `pizza`, `donut`, `cake`, `chair`, `couch`,
`potted plant`, `bed`, `dining table`, `toilet`, `tv`, `laptop`, `mouse`,
`remote`, `keyboard`, `cell phone`, `microwave`, `oven`, `toaster`, `sink`,
`refrigerator`, `book`, `clock`, `vase`, `scissors`, `teddy bear`, `hair drier`,
`toothbrush`.

## Model Backends

| Backend | How to select it | Intended use | Important limitation |
| --- | --- | --- | --- |
| Faster R-CNN MobileNetV3, pretrained on COCO | Default | Object-presence checks using CUDA when available, otherwise CPU | Does not answer attributes, actions, relationships, or open-ended questions |
| CNN + BiLSTM | `VQA_MODEL_BACKEND=cnn_lstm` | Baseline and training experiments on VAQ 2.0 | Demo checkpoint is weak; its scores are not calibrated |
| LLaVA 1.5 7B | Standalone script only | Optional image-question experiment | Requires CUDA and is not wired into `/predict` |
| ViT + RoBERTa | Components only | Model experimentation | No training or API inference pipeline is included |

The detector downloads approximately 74 MB of pretrained weights from the
Torchvision model repository on first startup. Later runs use Torch's local
cache. Detection scores are not calibrated probabilities that the final answer
is correct. A higher threshold can reduce weak detections but may also miss
real objects.

## Requirements

- Python 3.11
- Windows, Linux, or macOS for the API and CNN + BiLSTM training
- Internet access the first time the default detector starts
- Docker Desktop or Docker Engine if using the container workflow
- A compatible NVIDIA CUDA GPU and working `bitsandbytes` installation only
  for the optional LLaVA script

The API and default object detector do not require CUDA. To use a GPU for
detector inference, the machine needs an NVIDIA GPU, a compatible driver, and a
CUDA-enabled PyTorch/Torchvision installation. If
`torch.cuda.is_available()` is false, the detector automatically uses CPU. See
the [PyTorch installation selector](https://pytorch.org/get-started/locally/)
for a build compatible with your machine. The optional LLaVA script explicitly
stops if CUDA is unavailable. `requirements-extra.txt` is optional and is not
installed by the normal API setup.

## Install Dependencies

From the repository root, create a virtual environment and install the project libraries:

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

On Linux or macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

This installs the core libraries such as PyTorch, FastAPI, Uvicorn, Pillow,
NumPy, and the test/dev tools needed for local development.

## Install And Run

Start the development server:

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

On Linux or macOS:

```bash
python -m uvicorn app.main:app --reload
```

The first startup downloads the detector weights. The server log reports
whether the detector loaded and uses CUDA or CPU. Wait for the backend to load,
then open
`http://localhost:8000/docs` in a browser.

### Try It In Swagger

1. Open `http://localhost:8000/docs`.
2. Expand `POST /predict`, then select **Try it out**.
3. Choose a JPEG, PNG, or WebP file in the `image` field.
4. In the `question` field, ask about one supported object, for example
  `Is there a dog in this image?` or `Có chó trong ảnh không?`.
5. Leave `X-API-Key` blank unless the server was started with `VQA_API_KEY` set.
6. Select **Execute** and inspect the response body.

> Important: the API does not accept arbitrary objects. It only recognizes
> supported COCO categories such as `dog`, `cat`, `person`, `car`, `bicycle`,
> `bird`, `bus`, and similar labels. If you ask about an object not in the list,
> the server returns HTTP 422 with the error "Could not find a supported object
> category in the question". For a working example, use a question such as
> `Is there a dog in this image?`.

The browser communicates only with the local FastAPI service; model inference
runs locally on the machine where the server is running.

## Dataset And Training

Training is for the **CNN + BiLSTM baseline**, not the default COCO detector.
The VAQ 2.0 dataset is not bundled and must be obtained separately. Put the
three split files and resized COCO images under the data root:

```text
data/
  vaq2.0.TrainImages.txt
  vaq2.0.DevImages.txt
  vaq2.0.TestImages.txt
  val2014-resised/
    COCO_val2014_000000000000.jpg
    ...
```

Each split contains image/question/yes-no-answer records. Image references may
include a question index, for example `image.jpg#12`; the loader removes this
suffix before opening the image. Training checks that referenced images exist,
skips and logs missing files, and requires both `yes` and `no` labels.

Run training from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.train_cnn_lstm --epochs 5
```

Training uses CPU unless PyTorch detects CUDA. The default image encoder is a
small CNN initialized from scratch. Add `--pretrained` to use pretrained
ResNet-18 image weights through `timm`; the weights download on first use and
require internet access. A pretrained encoder can help, but does not guarantee
strong VQA accuracy.

The training command also supports `--seed` for reproducibility, `--limit` for
a small data-path check, and `--resume PATH` to continue from an epoch
checkpoint. Run `python -m src.train_cnn_lstm --help` to see all options.

The default is 5 epochs. Training evaluates validation macro-F1 and stops early
after 3 epochs without improvement. It reports validation and test accuracy,
macro precision/recall/F1, majority-class accuracy, and a confusion matrix.
Use the full data splits to measure quality; the CNN baseline can still perform
poorly on unfamiliar images.

Training writes `cnn_lstm_epoch_<N>.pt` checkpoints, the best epoch checkpoint
`cnn_lstm_best.pt`, the final `cnn_lstm_model.pt`, `vocab.json`, `labels.json`,
and updated hashes in `manifest.json`. The default output is `artifacts/`.
Training replaces the existing CNN baseline checkpoint and its vocabulary and
labels there; it does not retrain or replace the default COCO detector.

For a short pipeline smoke run, point outputs to a separate directory so the
bundled baseline artifacts remain unchanged. `--limit 100` uses at most 100
training records and a fraction of the validation/test records. This only
checks that the training path runs; the resulting metrics do not measure model
quality.

```powershell
$env:VQA_ARTIFACTS_DIR = "$env:TEMP\vqa-smoke-artifacts"
.\.venv\Scripts\python.exe -m src.train_cnn_lstm --limit 100 --epochs 1
Remove-Item Env:VQA_ARTIFACTS_DIR
```

To resume, use a saved epoch checkpoint from the same dataset and compatible
model configuration. `--epochs` is the desired total epoch, not the number of
additional epochs:

```powershell
.\.venv\Scripts\python.exe -m src.train_cnn_lstm --epochs 10 --resume artifacts/cnn_lstm_epoch_3.pt
```

Set `VQA_DATA_DIR` to a different dataset root if needed. Set
`VQA_ARTIFACTS_DIR` to an existing or new writable directory to keep model
outputs separate from `artifacts/`. The artifacts directory is also where the
API looks for the CNN checkpoint when `VQA_MODEL_BACKEND=cnn_lstm`.

## API

The interactive API docs are at `http://localhost:8000/docs`. Example request
from Windows PowerShell:

```powershell
curl.exe -X POST http://localhost:8000/predict `
  -F "image=@example.jpg" `
  -F "question=Is there a dog in this image?"
```

Example response:

```json
{
  "answer": "yes",
  "target": "dog",
  "detection_score": 0.91,
  "threshold": 0.5,
  "model": "Faster R-CNN MobileNetV3 320 (COCO)",
  "detected_objects": [{"label": "dog", "score": 0.91}]
}
```

The values above are illustrative; scores depend on the image. `detected_objects`
lists all detected objects whose score meets the threshold, not just the
requested target. `detection_score` is the highest score for `target`, even if
it is below the threshold. If there is no detection for the target, the score
is `0` and `answer` is `no`.

The detector finds one supported category in the question. Use one category per
question, such as `dog`, `cat`, `person`, `car`, or `bicycle`. A few common
Vietnamese names are mapped to COCO names, including `chó`, `mèo`, `người`,
`xe đạp`, `xe hơi`, `ô tô`, `xe máy`, `xe buýt`, and `máy bay`. If no supported
category is found, the API returns HTTP 422. Questions about color, activity,
count, or relations between objects are not supported. When a question mentions
multiple object classes, the parser selects the longest recognized class name;
that is not a semantic understanding of the question. Questions asking for a
count or using negation can still be reduced to a simple presence check; phrase
questions as affirmative existence checks to avoid ambiguity.

`VQA_DETECTION_THRESHOLD` controls the yes/no cutoff. It must be between `0`
and `1`; the default is `0.5`. Raising it makes the detector more conservative,
but can increase missed objects. The score is not a calibrated probability
that the answer is correct.

To compare against the original CNN + BiLSTM baseline, set the backend before
starting the API:

```powershell
$env:VQA_MODEL_BACKEND = "cnn_lstm"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

The baseline returns a different response, with `confidence` and
`probabilities` fields. Those values come from the CNN + BiLSTM classifier and
are not the detector score. For example:

```json
{
  "answer": "no",
  "confidence": 0.57,
  "probabilities": {"no": 0.57, "yes": 0.43}
}
```

### API Endpoints And Errors

- `/health` reports model status and returns HTTP 200 even if the model is not loaded.
- `/live` is a process liveness check.
- `/ready` returns HTTP 503 until the model is loaded; container healthchecks use this endpoint.
- `POST /predict` accepts multipart form fields `image` and `question`.

Common `/predict` errors:

| Status | Meaning |
| --- | --- |
| 400 | The uploaded file could not be decoded as an image |
| 401 | `VQA_API_KEY` is set and the `X-API-Key` header is missing or incorrect |
| 415 | The upload content type is not JPEG, PNG, or WebP |
| 422 | The question is empty or has no supported object category |
| 503 | The prediction backend did not load; check server logs and `/ready` |

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `VQA_MODEL_BACKEND` | `detector` | Select `detector` or `cnn_lstm` |
| `VQA_DETECTION_THRESHOLD` | `0.5` | Yes/no score threshold for the detector, from 0 to 1 |
| `VQA_API_KEY` | unset | If set, require a matching `X-API-Key` request header |
| `VQA_ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `VQA_DATA_DIR` | `./data` | Root directory containing training splits and images |
| `VQA_ARTIFACTS_DIR` | `./artifacts` | Directory containing/writing CNN baseline artifacts |
| `VQA_PRETRAINED` | `0` | Enable a pretrained CNN image backbone for training when set to `1` |

Set environment variables before starting the server. For production, replace
the default wildcard CORS origin with trusted origins and apply rate limiting at
an API gateway or reverse proxy. Responses include an `X-Request-ID`; request
status and duration are logged.

## Tests And Lint

Run the checks from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\ruff.exe check .
```

The test suite covers API contracts, detector target parsing and thresholds,
CNN model output, training metrics, dataset split parsing, and vocabulary
behavior. Detector unit tests use a fake model and do not download weights. A
real training run is not part of the automated tests; use the isolated smoke
training command above to verify data loading through checkpoint saving.

GitHub Actions runs Ruff, pytest, and a Docker image build for pushes and pull
requests targeting `main`. A push to `main` also publishes the image to GitHub
Container Registry when repository package publishing is enabled.

To exercise the running HTTP service with a local image, start Uvicorn first,
then run:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test.py .\dog.jpg "Is there a dog in this image?"
```

The smoke script requires `httpx` from `requirements-dev.txt` and a reachable
server at `http://localhost:8000`.

## Docker

```powershell
docker build -t vqa-api:local .
docker run --rm -p 8000:8000 vqa-api:local
```

The image includes CPU API dependencies and the CNN baseline artifacts, but not
the detector weights, optional transformer/VLM packages, or training dataset.
The Dockerfile installs CPU-only PyTorch/Torchvision, so the container uses the
CPU even if its host has a GPU. For CUDA inference, run the API directly in a
CUDA-enabled Python environment. The detector downloads its weights when the
container first starts, so it needs outbound internet access. Wait for the
container health status to become `healthy`, then open
`http://localhost:8000/docs`.

To adjust the detector threshold in Docker:

```powershell
docker run --rm -p 8000:8000 -e VQA_DETECTION_THRESHOLD=0.6 vqa-api:local
```

## Optional LLaVA

This standalone experiment is separate from the API. It reads the selected
sample from the VAQ 2.0 test split and its image from `VQA_DATA_DIR`; therefore
the dataset must be present. Run it on Linux with a compatible NVIDIA CUDA GPU
and working `bitsandbytes` support:

```bash
python -m pip install -r requirements-extra.txt
python -m src.infer_vlm --index 0
```

The script loads `llava-hf/llava-1.5-7b-hf` by default and downloads model
files on first use. Override it with `--model-id`. It is not connected to
`/predict`; selecting `VQA_MODEL_BACKEND=cnn_lstm` does not select LLaVA.

## Repository Layout

```text
app/                 FastAPI routes, lifecycle, and response schemas
src/detector.py      Default COCO detector; auto-selects CUDA or CPU
src/predictor.py     CNN + BiLSTM checkpoint inference backend
src/train_cnn_lstm.py
                     Training entry point for the CNN + BiLSTM baseline
src/data/            VAQ split parser, dataset, image transforms, vocabulary
src/models/          CNN + BiLSTM and ViT + RoBERTa model definitions
tests/               API, detector, training-engine, data, and model tests
scripts/smoke_test.py
                     Manual HTTP smoke test for a running API
artifacts/           Bundled CNN baseline checkpoint and metadata
data/                Local dataset location; dataset files are not committed
```

## Artifacts And Limitations

`artifacts/manifest.json` records SHA-256 checksums for the CNN + BiLSTM
checkpoint, vocabulary, and labels. The baseline predictor verifies the model
checkpoint checksum before loading it. Training regenerates the hashes after
writing new artifacts.

Neither backend guarantees correct answers. The detector is limited to its
COCO object classes; the baseline is a small binary classifier and its scores
are not calibrated. Evaluate on representative, held-out images before relying
on predictions. Do not commit private images, API keys, or datasets. For large
model files, use Git LFS, a model registry, or object storage instead of normal
Git history.
