"""
Download MS COCO val2017 dataset and build metadata.csv.

This script:
1. Downloads val2017.zip (5,000 images, ~1 GB)
2. Downloads annotations_trainval2017.zip (captions + instances)
3. Extracts only the needed files
4. Builds metadata.csv with columns: id, image_path, caption, category

Usage:
    python scripts/download_dataset.py
"""

import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# Add project root to path so we can import config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR,
    IMAGES_DIR,
    ANNOTATIONS_DIR,
    METADATA_PATH,
    COCO_IMAGES_URL,
    COCO_ANNOTATIONS_URL,
)


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """
    Download a file from a URL with a progress bar.

    Skips download if the file already exists.
    """
    if dest_path.exists():
        print(f"  [skip] Already exists: {dest_path}")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {url}")
    print(f"  Destination: {dest_path}")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    progress = tqdm(total=total_size, unit="B", unit_scale=True, desc="  Progress")

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                progress.update(len(chunk))
    progress.close()
    print(f"  Done: {dest_path} ({dest_path.stat().st_size / 1e6:.1f} MB)")


def extract_images(zip_path: Path, dest_dir: Path) -> None:
    """
    Extract images from val2017.zip into dest_dir/images/.

    The zip contains a val2017/ folder with JPEG files. We extract the images
    directly into dest_dir/ (stripping the val2017/ prefix).
    """
    if dest_dir.exists() and any(dest_dir.iterdir()):
        existing = len(list(dest_dir.glob("*.jpg")))
        if existing >= 5000:
            print(f"  [skip] Images already extracted: {existing} files in {dest_dir}")
            return

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting images to {dest_dir} ...")

    with zipfile.ZipFile(zip_path, "r") as z:
        members = [m for m in z.namelist() if m.startswith("val2017/") and m.endswith(".jpg")]
        for member in tqdm(members, desc="  Extracting"):
            # Strip the val2017/ prefix — extract directly into dest_dir
            filename = os.path.basename(member)
            target = dest_dir / filename
            with z.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())

    count = len(list(dest_dir.glob("*.jpg")))
    print(f"  Extracted {count} images.")


def extract_annotations(zip_path: Path, dest_dir: Path) -> None:
    """
    Extract captions_val2017.json and instances_val2017.json from the
    annotations zip file.
    """
    needed = ["captions_val2017.json", "instances_val2017.json"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    all_present = all((dest_dir / f).exists() for f in needed)
    if all_present:
        print(f"  [skip] Annotations already extracted in {dest_dir}")
        return

    print(f"  Extracting annotations to {dest_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            basename = os.path.basename(member)
            if basename in needed:
                target = dest_dir / basename
                with z.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                print(f"    -> {target}")

    print("  Annotations extracted.")


def build_metadata(annotations_dir: Path, images_dir: Path, output_path: Path) -> pd.DataFrame:
    """
    Build metadata.csv from COCO annotation files.

    Each row:
        id          — COCO image_id
        image_path  — relative path under data/images/
        caption     — first caption for that image
        category    — most frequent object category (or "unknown")
    """
    print("\n  Building metadata.csv ...")

    # --- Load captions ---
    captions_file = annotations_dir / "captions_val2017.json"
    with open(captions_file, "r", encoding="utf-8") as f:
        captions_data = json.load(f)

    # Map image_id -> first caption (sorted by annotation id for determinism)
    caption_annots = sorted(captions_data["annotations"], key=lambda a: a["id"])
    image_captions = {}
    for ann in caption_annots:
        img_id = ann["image_id"]
        if img_id not in image_captions:
            image_captions[img_id] = ann["caption"]

    print(f"    Captions loaded: {len(image_captions)} images have captions.")

    # --- Load instances (object categories) ---
    instances_file = annotations_dir / "instances_val2017.json"
    with open(instances_file, "r", encoding="utf-8") as f:
        instances_data = json.load(f)

    # Build category ID -> name mapping
    cat_id_to_name = {}
    for cat in instances_data["categories"]:
        cat_id_to_name[cat["id"]] = cat["name"]

    # Map image_id -> most frequent category
    image_categories_counter = {}
    for ann in instances_data["annotations"]:
        img_id = ann["image_id"]
        cat_name = cat_id_to_name.get(ann["category_id"], "unknown")
        if img_id not in image_categories_counter:
            image_categories_counter[img_id] = Counter()
        image_categories_counter[img_id][cat_name] += 1

    image_category = {}
    for img_id, counter in image_categories_counter.items():
        # Most common category (by annotation count)
        image_category[img_id] = counter.most_common(1)[0][0]

    print(f"    Categories loaded: {len(image_category)} images have object annotations.")

    # --- Build image list from annotation metadata ---
    # Use the images list from captions (it contains all 5,000 val images)
    image_list = captions_data["images"]

    rows = []
    missing_images = 0
    for img_info in image_list:
        img_id = img_info["id"]
        filename = img_info["file_name"]
        image_path = f"data/images/{filename}"

        # Verify image file exists on disk
        full_path = images_dir / filename
        if not full_path.exists():
            missing_images += 1
            continue

        caption = image_captions.get(img_id, "")
        category = image_category.get(img_id, "unknown")

        rows.append({
            "id": img_id,
            "image_path": image_path,
            "caption": caption,
            "category": category,
        })

    if missing_images > 0:
        print(f"    WARNING: {missing_images} images listed in annotations but not found on disk.")

    df = pd.DataFrame(rows)
    df = df.sort_values("id").reset_index(drop=True)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"    Metadata saved: {output_path}")
    print(f"    Total rows:     {len(df)}")
    print(f"    Columns:        {list(df.columns)}")
    print(f"    Categories:     {df['category'].nunique()} unique")
    print(f"    Sample:")
    print(df.head(3).to_string(index=False))

    return df


def verify_dataset(metadata_path: Path, images_dir: Path) -> None:
    """Quick verification that the dataset is complete and consistent."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    df = pd.read_csv(metadata_path)
    n_rows = len(df)
    n_images = len(list(images_dir.glob("*.jpg")))

    print(f"  metadata.csv rows:      {n_rows}")
    print(f"  Images on disk:         {n_images}")
    print(f"  IDs unique:             {df['id'].nunique()} / {n_rows}")
    print(f"  Captions non-empty:     {(df['caption'].str.len() > 0).sum()} / {n_rows}")
    print(f"  Categories non-unknown: {(df['category'] != 'unknown').sum()} / {n_rows}")
    print(f"  Top categories:")
    print(df["category"].value_counts().head(10).to_string())

    # Spot-check: verify a few image paths
    sample = df.sample(min(5, len(df)), random_state=42)
    all_ok = True
    for _, row in sample.iterrows():
        # image_path is relative like "data/images/xxx.jpg"
        full = images_dir / Path(row["image_path"]).name
        if not full.exists():
            print(f"  MISSING: {full}")
            all_ok = False

    if n_rows >= 4990 and all_ok:
        print("\n  ✓ Dataset verification PASSED")
    else:
        print("\n  ✗ Dataset verification FAILED — check logs above")


def main():
    print("=" * 60)
    print("MS COCO val2017 — Download & Metadata Builder")
    print("=" * 60)

    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download images
    print("\n[1/4] Downloading val2017 images ...")
    images_zip = DATA_DIR / "val2017.zip"
    download_file(COCO_IMAGES_URL, images_zip)

    # Step 2: Download annotations
    print("\n[2/4] Downloading annotations ...")
    annotations_zip = DATA_DIR / "annotations_trainval2017.zip"
    download_file(COCO_ANNOTATIONS_URL, annotations_zip)

    # Step 3: Extract
    print("\n[3/4] Extracting files ...")
    extract_images(images_zip, IMAGES_DIR)
    extract_annotations(annotations_zip, ANNOTATIONS_DIR)

    # Step 4: Build metadata
    print("\n[4/4] Building metadata ...")
    build_metadata(ANNOTATIONS_DIR, IMAGES_DIR, METADATA_PATH)

    # Verify
    verify_dataset(METADATA_PATH, IMAGES_DIR)

    print("\n" + "=" * 60)
    print("Dataset setup complete!")
    print(f"  Images:   {IMAGES_DIR}")
    print(f"  Metadata: {METADATA_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
