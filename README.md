# Visual Question Answering (Yes/No)

This project provides three approaches to binary visual question answering:

- A lightweight CNN + BiLSTM baseline used by the FastAPI service.
- An optional ViT + RoBERTa model for transformer-based experiments.
- Optional zero-shot LLaVA inference for GPU-based exploration.

The service accepts an image and a yes/no question, then returns the predicted
answer and class probabilities.

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -v
uvicorn app.main:app --reload
```

The repository includes a small demo checkpoint under `artifacts/`, so the API
can start without training or downloading a backbone. The API is available at
`http://localhost:8000/docs`.

## Training

Download the VAQ 2.0 split files and resized images into `data/`:

```text
data/
  vaq2.0.TrainImages.txt
  vaq2.0.DevImages.txt
  vaq2.0.TestImages.txt
  val2014-resised/
```

Then run:

```bash
python -m src.train_cnn_lstm
# Use --limit for a small smoke run
python -m src.train_cnn_lstm --limit 100 --epochs 1
```

Set `VQA_DATA_DIR` and `VQA_ARTIFACTS_DIR` when the data or model files are
stored elsewhere. The training command writes the vocabulary, labels and
checkpoint to the artifacts directory.

## API example

```bash
curl -X POST http://localhost:8000/predict ^
  -F "image=@example.jpg" ^
  -F "question=Is this image red?"
```

Example response:

```json
{
  "answer": "yes",
  "confidence": 0.87,
  "probabilities": {"yes": 0.87, "no": 0.13}
}
```

`/health` reports whether the model was loaded. For production deployments,
use a separate readiness check at the reverse proxy/orchestrator layer and
configure CORS to allow only trusted origins. Set `VQA_ALLOWED_ORIGINS` to a
comma-separated list of origins and `VQA_API_KEY` to enable the optional
`X-API-Key` check. Every response includes an `X-Request-ID`, and request
duration/status are logged for basic observability. Put rate limiting at an API
gateway or reverse proxy so it remains correct across multiple workers.

## Docker

```bash
docker build -t vqa-api:local .
docker run --rm -p 8000:8000 vqa-api:local
```

The image contains only the CPU API dependencies. Optional transformer/VLM
dependencies are listed separately in `requirements-extra.txt`.

## Optional pipelines

Install the optional dependencies before using the transformer or LLaVA code:

```bash
pip install -r requirements-extra.txt
python -m src.infer_vlm --index 0
```

LLaVA inference requires a compatible CUDA GPU and bitsandbytes installation;
the 7B model is not supported by the CPU fallback.

## Artifacts and reproducibility

The demo checkpoint is committed for an out-of-the-box API demo. Its checksums
and provenance are recorded in `artifacts/manifest.json`. For larger or
frequently updated models, use Git LFS, a model registry, or object storage
instead of committing binaries to normal Git history. When replacing an
artifact, update the manifest and record the training configuration and
dataset version.

## Limitations

This is a binary yes/no baseline. It is not a general-purpose VQA system, and
its confidence is not calibrated. Report validation/test metrics for each
trained checkpoint before using it for decisions or public deployment.
