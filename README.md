# Visual Question Answering (Yes/No)

See project docs / code comments. (Full README intentionally left for you to write.)

Quickstart:
    pip install -r requirements-dev.txt
    python -m src.train_cnn_lstm          # place dataset under data/ first
    uvicorn app.main:app --reload         # serves the trained checkpoint
    pytest -v
    docker build -t vqa-api . && docker run -p 8000:8000 vqa-api
