"""Quick manual smoke test against a running API instance.

Usage:
    python scripts/smoke_test.py path/to/image.jpg "Is this a dog?"
"""
import sys

import requests


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python scripts/smoke_test.py <image_path> <question>")
        raise SystemExit(1)

    image_path, question = sys.argv[1], sys.argv[2]
    with open(image_path, "rb") as f:
        resp = requests.post(
            "http://localhost:8000/predict",
            files={"image": f},
            data={"question": question},
            timeout=30,
        )
    resp.raise_for_status()
    print(resp.json())


if __name__ == "__main__":
    main()
