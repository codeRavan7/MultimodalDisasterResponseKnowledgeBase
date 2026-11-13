"""Step 1: download the CrisisMMD v2.0 annotations and the images for the
chosen events from QCRI's official Hugging Face copy of the dataset.

Dataset: Alam, Ofli & Imran, "CrisisMMD: Multimodal Twitter Datasets from
Natural Disasters", ICWSM 2018. License CC BY-NC-SA 4.0 - keep data/ private.
"""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE_URL = "https://huggingface.co/datasets/QCRI/CrisisMMD/resolve/main/"
DATA_DIR = Path(__file__).parent / "data" / "CrisisMMD_v2.0"

# A laptop-sized subset: three different hazard types, about 4,000 images.
EVENTS = ["california_wildfires", "mexico_earthquake", "srilanka_floods"]
TASKS = ["informative", "humanitarian", "damage"]
SPLITS = ["train", "dev", "test"]


def fetch(relative_path):
    target = DATA_DIR / relative_path
    if target.exists():
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(BASE_URL + relative_path, target)
        return True
    except Exception as err:
        print(f"  failed: {relative_path} ({err})")
        return False


def main():
    print("Downloading annotation files...")
    for task in TASKS:
        for split in SPLITS:
            fetch(f"{task}/{split}.json")

    # Every image appears in the 'informative' task, so it lists them all.
    image_paths = set()
    for split in SPLITS:
        rows = json.loads((DATA_DIR / "informative" / f"{split}.json").read_text(encoding="utf-8"))
        image_paths.update(r["image_path"] for r in rows if r["event_name"] in EVENTS)

    print(f"Downloading {len(image_paths)} images for {', '.join(EVENTS)}...")
    with ThreadPoolExecutor(max_workers=16) as pool:
        ok = sum(pool.map(fetch, sorted(image_paths)))
    print(f"Done: {ok}/{len(image_paths)} images in {DATA_DIR}")


if __name__ == "__main__":
    main()
