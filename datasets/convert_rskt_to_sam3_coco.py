#!/usr/bin/env python3
"""
Convert RSKT-Seg DLRSD/iSAID semantic splits into SAM3 image-NP JSON format.

Output JSON follows the SAM3 "silver" style:
  - each (image, class-query) is one item in `images`
  - each positive query has one semantic annotation in `annotations`
  - RLE mask is stored in COCO format

This script enforces:
  - split by GT pixels (not only by multi-label metadata)
  - one query per class present in each image GT
  - guaranteed mask annotation for every exported query
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from pycocotools import mask as mask_utils
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pycocotools is required. Install it with `pip install pycocotools`."
    ) from exc


@dataclass(frozen=True)
class Category:
    id: int
    name: str
    color: Tuple[int, int, int]


# DLRSD class/RGB table (transcribed from dataset description and register_DLRSD.py)
DLRSD_CATEGORIES: List[Category] = [
    Category(1, "airplane", (166, 202, 240)),
    Category(2, "bare soil", (128, 128, 0)),
    Category(3, "buildings", (0, 0, 128)),
    Category(4, "cars", (255, 0, 0)),
    Category(5, "chaparral", (0, 128, 0)),
    Category(6, "court", (128, 0, 0)),
    Category(7, "dock", (255, 233, 233)),
    Category(8, "field", (160, 160, 164)),
    Category(9, "grass", (0, 128, 128)),
    Category(10, "mobile home", (90, 87, 255)),
    Category(11, "pavement", (255, 255, 0)),
    Category(12, "sand", (255, 192, 0)),
    Category(13, "sea", (0, 0, 255)),
    Category(14, "ship", (255, 0, 192)),
    Category(15, "tanks", (128, 0, 128)),
    Category(16, "trees", (0, 255, 0)),
    Category(17, "water", (0, 255, 255)),
]

# iSAID class/RGB table from register_iSAID.py
ISAID_CATEGORIES: List[Category] = [
    Category(1, "ship", (0, 0, 63)),
    Category(2, "storage tank", (0, 63, 63)),
    Category(3, "baseball diamond", (0, 63, 0)),
    Category(4, "tennis court", (0, 63, 127)),
    Category(5, "basketball court", (0, 63, 191)),
    Category(6, "ground track field", (0, 63, 255)),
    Category(7, "bridge", (0, 127, 63)),
    Category(8, "large vehicle", (0, 127, 127)),
    Category(9, "small vehicle", (0, 0, 127)),
    Category(10, "helicopter", (0, 0, 191)),
    Category(11, "swimming pool", (0, 0, 255)),
    Category(12, "roundabout", (0, 191, 127)),
    Category(13, "soccer ball field", (0, 127, 191)),
    Category(14, "plane", (0, 127, 255)),
    Category(15, "harbor", (0, 100, 155)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DLRSD/iSAID splits to SAM3 semantic JSON."
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("datasets"),
        help="Root path that contains DLRSD_split/iSAID_split.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets") / "sam3_semantic_fixed",
        help="Output root for SAM3 JSON files.",
    )
    parser.add_argument(
        "--dataset",
        choices=["dlrsd", "isaid", "all"],
        default="all",
        help="Which dataset to convert.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Split names to convert.",
    )
    return parser.parse_args()


def normalize_mask_stem(stem: str) -> str:
    suffix = "_instance_color_RGB"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def index_masks(mask_dirs: Sequence[Path]) -> Dict[str, Path]:
    # Earlier directories in mask_dirs have higher priority.
    out: Dict[str, Path] = {}
    for mask_dir in mask_dirs:
        if not mask_dir.exists():
            continue
        for mask_path in sorted(mask_dir.glob("*.png")):
            stem = normalize_mask_stem(mask_path.stem)
            if stem not in out:
                out[stem] = mask_path
    return out


def list_image_mask_pairs(image_dir: Path, mask_dirs: Sequence[Path]) -> List[Tuple[Path, Path]]:
    mask_index = index_masks(mask_dirs)
    pairs: List[Tuple[Path, Path]] = []

    image_files = sorted(
        [
            p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        ]
    )
    for img_path in image_files:
        mask_path = mask_index.get(img_path.stem)
        if mask_path is not None:
            pairs.append((img_path, mask_path))
    return pairs


def pick_existing_subdir(base: Path, candidates: Sequence[str]) -> Path:
    for name in candidates:
        p = base / name
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(f"None of {list(candidates)} exists in {base}")


def load_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as img:
        arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def preferred_id_mode(dataset_name: str, mask_path: Path) -> Optional[str]:
    parent_name = mask_path.parent.name.lower()
    if dataset_name == "dlrsd":
        return "minus1" if parent_name == "d2masks" else "direct"
    if dataset_name == "isaid":
        return "minus1"
    return None


def decode_from_single_channel(
    label: np.ndarray, class_ids: Sequence[int], mode: Optional[str]
) -> Dict[int, np.ndarray]:
    label = label.astype(np.int32, copy=False)
    unique_set = set(int(v) for v in np.unique(label))

    if mode is None:
        direct_score = sum(1 for cid in class_ids if cid in unique_set)
        minus1_score = sum(1 for cid in class_ids if (cid - 1) in unique_set)
        mode = "minus1" if minus1_score > direct_score else "direct"

    out: Dict[int, np.ndarray] = {}
    for cid in class_ids:
        value = cid - 1 if mode == "minus1" else cid
        mask = label == value
        if mask.any():
            out[cid] = mask
    return out


def extract_class_binary_masks(
    raw_mask: np.ndarray, categories: Sequence[Category], id_mode: Optional[str]
) -> Dict[int, np.ndarray]:
    class_ids = [c.id for c in categories]

    # RGB mask mode
    if raw_mask.ndim == 3 and raw_mask.shape[2] >= 3:
        rgb = raw_mask[:, :, :3].astype(np.int32, copy=False)
        # If all channels equal, treat as single-channel label.
        if np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(
            rgb[:, :, 1], rgb[:, :, 2]
        ):
            return decode_from_single_channel(rgb[:, :, 0], class_ids, id_mode)

        out: Dict[int, np.ndarray] = {}
        for c in categories:
            color = np.array(c.color, dtype=np.int32).reshape(1, 1, 3)
            m = np.all(rgb == color, axis=2)
            if m.any():
                out[c.id] = m
        if out:
            return out

        # Fallback if RGB direct matching did not find any class.
        return decode_from_single_channel(rgb[:, :, 0], class_ids, id_mode)

    if raw_mask.ndim == 2:
        return decode_from_single_channel(raw_mask, class_ids, id_mode)

    raise ValueError(f"Unsupported mask shape: {raw_mask.shape}")


def encode_binary_mask(mask: np.ndarray) -> Tuple[Dict[str, object], List[float], float]:
    """
    Return COCO-style segmentation / bbox / area:
      - segmentation: compressed RLE
      - bbox: absolute xywh in pixels
      - area: absolute pixel area
    """
    mask_u8 = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask_u8)

    bbox = [float(x) for x in mask_utils.toBbox(rle).tolist()]   # absolute xywh
    area = float(mask_utils.area(rle))                            # absolute pixel area

    rle["counts"] = rle["counts"].decode("ascii")
    return rle, bbox, area


def convert_split(
    dataset_name: str,
    split: str,
    image_dir: Path,
    mask_dirs: Sequence[Path],
    categories: Sequence[Category],
    output_json: Path,
) -> None:
    pairs = list_image_mask_pairs(image_dir, mask_dirs)
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found in split: {image_dir}")

    images: List[Dict[str, object]] = []
    annotations: List[Dict[str, object]] = []
    queries: List[Dict[str, object]] = []

    ann_id = 1
    image_np_id = 1
    query_id = 1

    cat_id_to_name = {c.id: c.name for c in categories}

    for img_path, mask_path in pairs:
        with Image.open(img_path) as img:
            width, height = img.size

        raw_mask = load_mask(mask_path)
        id_mode = preferred_id_mode(dataset_name, mask_path)
        class_masks = extract_class_binary_masks(raw_mask, categories, id_mode)

        for class_id in sorted(class_masks.keys()):
            class_mask = class_masks[class_id]
            if not class_mask.any():
                continue
            if int(class_mask.sum()) < 1:
                continue

            image_item = {
                "id": image_np_id,
                "file_name": img_path.name,
                "width": int(width),
                "height": int(height),
                "text_input": cat_id_to_name[class_id],
                "queried_category": str(class_id),
                "is_instance_exhaustive": 0,
                "is_pixel_exhaustive": 1,
            }
            images.append(image_item)

            rle, bbox, area = encode_binary_mask(class_mask)

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_np_id,
                    "category_id": class_id,
                    "source": "manual",
                    "iscrowd": 0,
                    "area": area,
                    "bbox": bbox,
                    "segmentation": rle,
                }
            )

            queries.append(
                {
                    "id": query_id,
                    "image_id": image_np_id,
                    "query_text": cat_id_to_name[class_id],
                    "object_ids_output": [ann_id],
                    "is_exhaustive": True,
                    "is_pixel_exhaustive": True,
                    "query_processing_order": 0,
                    "original_cat_id": class_id,
                    "ptr_x_query_id": None,
                    "ptr_y_query_id": None,
                    "input_box": None,
                    "input_box_label": None,
                    "input_points": None,
                }
            )

            ann_id += 1
            query_id += 1
            image_np_id += 1

    categories_json = [
        {
            "id": c.id,
            "name": c.name,
            "supercategory": "semantic",
            "color": list(c.color),
        }
        for c in categories
    ]

    output = {
        "images": images,
        "annotations": annotations,
        "categories": categories_json,
        "queries": queries,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    print(
        f"[{dataset_name}:{split}] "
        f"pairs={len(pairs)}, image_nps={len(images)}, anns={len(annotations)}, "
        f"queries={len(queries)} -> {output_json}"
    )


def convert_dlrsd(datasets_root: Path, output_root: Path, splits: Iterable[str]) -> None:
    split_root = datasets_root / "DLRSD_split"
    for split in splits:
        split_dir = split_root / split
        image_dir = pick_existing_subdir(split_dir, ["imgs", "images"])
        mask_dirs = [split_dir / "masks", split_dir / "D2masks"]
        out_json = output_root / "dlrsd" / f"{split}.sam3.semantic.json"
        convert_split("dlrsd", split, image_dir, mask_dirs, DLRSD_CATEGORIES, out_json)


def convert_isaid(datasets_root: Path, output_root: Path, splits: Iterable[str]) -> None:
    split_root = datasets_root / "iSAID_split"
    for split in splits:
        split_dir = split_root / split
        image_dir = pick_existing_subdir(split_dir, ["imgs", "images"])
        mask_dirs = [split_dir / "masks", split_dir / "D2masks"]
        out_json = output_root / "isaid" / f"{split}.sam3.semantic.json"
        convert_split("isaid", split, image_dir, mask_dirs, ISAID_CATEGORIES, out_json)


def main() -> None:
    args = parse_args()
    datasets_root = args.datasets_root
    output_root = args.output_root
    splits = args.splits

    if args.dataset in {"dlrsd", "all"}:
        convert_dlrsd(datasets_root, output_root, splits)
    if args.dataset in {"isaid", "all"}:
        convert_isaid(datasets_root, output_root, splits)


if __name__ == "__main__":
    main()


