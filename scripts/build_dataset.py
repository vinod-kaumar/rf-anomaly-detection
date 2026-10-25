# FILE: scripts/build_dataset.py
# What this file does:
#   - Generates the train / validation / test splits once and saves them to
#     data/<split>.npz, so notebooks load them in seconds instead of
#     regenerating them.
#   - Each file holds: x (complex64 windows), y (0 normal, 1 anomalous, -1
#     ambiguous), family, snr_db, rec_id.
#   - Default sizes: train 300 recordings (all normal), val 150, test 350.
#     With 15 windows per recording this is 4,500 / 2,250 / 5,250 windows.
#   - Window length 4096, hop 2048, ambiguity rule 50% (see windowing.py).
#   - Run from the project root:  python scripts/build_dataset.py
#   - Use --small for a quick test run (about 1/10 of the size).

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.splits import build_split          # noqa: E402
from src.dsp.windowing import collect_windows    # noqa: E402

WIN, HOP = 4096, 2048
SIZES = {"train": 300, "val": 150, "test": 350}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="build about 1/10 of the data")
    parser.add_argument("--out", default="data", help="output folder")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for split, n_rec in SIZES.items():
        n_rec = max(7, n_rec // 10) if args.small else n_rec
        t0 = time.time()
        recs = build_split(split, n_rec)
        d = collect_windows(recs, WIN, HOP)
        np.savez(out / f"{split}.npz", **d)
        y = d["y"]
        print(f"{split:5s}: {n_rec:4d} recordings -> {len(y):5d} windows "
              f"(normal {np.sum(y == 0)}, anomalous {np.sum(y == 1)}, ambiguous {np.sum(y == -1)}) "
              f"in {time.time() - t0:.1f}s -> {out / (split + '.npz')}")


if __name__ == "__main__":
    main()