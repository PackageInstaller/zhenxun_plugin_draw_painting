"""Manual GPU probe: no bot hooks, image writes or production database access.

Run each tagger in a separate process to avoid keeping duplicate model sessions.
Usage: python tests/benchmark_tagger.py wd|camie path/to/image.png
"""

# ruff: noqa: T201 - manual diagnostic output

import importlib
from pathlib import Path
import sys
import time

import test_subject_features  # noqa: F401 - isolated package bootstrap


def main():
    module = importlib.import_module("_painting_tests.services.vision.taggers")
    runtime = importlib.import_module("_painting_tests.services.vision.runtime")
    model = (module.WDTaggerModel if sys.argv[1] == "wd" else module.CamieTaggerModel)()
    image = runtime.open_rgb(Path(sys.argv[2]))
    print("provider", model.session.get_providers(), flush=True)
    model.predict_images([image])  # warmup
    for batch in (1, 8):
        model.batch_size = batch
        start = time.perf_counter()
        results = model.predict_images([image] * 8)
        elapsed = time.perf_counter() - start
        errors = [str(value) for value in results if isinstance(value, Exception)]
        print(
            {
                "batch": batch,
                "effective_batch": model.batch_size,
                "seconds": round(elapsed, 3),
                "images_per_second": round(8 / elapsed, 3),
                "errors": errors,
            },
            flush=True,
        )
        if errors:
            raise RuntimeError("Inference failed")


if __name__ == "__main__":
    main()
