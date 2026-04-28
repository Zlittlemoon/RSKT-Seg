import os
import cv2
import numpy as np


DLRSD_CATEGORIES = [
    {"color": [166, 202, 240], "id": 1, "name": "airplane"},
    {"color": [128, 128, 0], "id": 2, "name": "bare soil"},
    {"color": [0, 0, 128], "id": 3, "name": "buildings"},
    {"color": [255, 0, 0], "id": 4, "name": "cars"},
    {"color": [0, 128, 0], "id": 5, "name": "chaparral"},
    {"color": [128, 0, 0], "id": 6, "name": "court"},
    {"color": [255, 233, 233], "id": 7, "name": "dock"},
    {"color": [160, 160, 164], "id": 8, "name": "field"},
    {"color": [0, 128, 128], "id": 9, "name": "grass"},
    {"color": [90, 87, 255], "id": 10, "name": "mobile home"},
    {"color": [255, 255, 0], "id": 11, "name": "pavement"},
    {"color": [255, 192, 0], "id": 12, "name": "sand"},
    {"color": [0, 0, 255], "id": 13, "name": "sea"},
    {"color": [255, 0, 192], "id": 14, "name": "ship"},
    {"color": [128, 0, 128], "id": 15, "name": "tanks"},
    {"color": [0, 255, 0], "id": 16, "name": "trees"},
    {"color": [0, 255, 255], "id": 17, "name": "water"},
]


def normalize_to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def to_single_channel(label: np.ndarray) -> np.ndarray:
    if label.ndim == 2:
        return label
    # For RGBA/RGB masks where channels are identical, use the first channel.
    return label[..., 0]


def get_dlrds_meta():
    # Keep the same metadata structure as register_DLRSD.py
    stuff_ids = [k["id"] for k in DLRSD_CATEGORIES]  # dataset ids: 1..17
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}  # 0..16
    stuff_colors = [k["color"] for k in DLRSD_CATEGORIES]  # index by contiguous id
    return stuff_dataset_id_to_contiguous_id, stuff_colors


def colorize_with_visualizer_logic(
    raw_label: np.ndarray, raw_ids_are_dataset_ids: bool
) -> np.ndarray:
    """
    Mimic Detectron2 Visualizer.draw_sem_seg logic for metadata colors:
    - colors are indexed by contiguous class id (0..C-1)
    - if input label is dataset id (1..17), map it via stuff_dataset_id_to_contiguous_id first
    - if input label is already contiguous id (0..16), use it directly
    """
    label = to_single_channel(raw_label).astype(np.int32)
    id_map, stuff_colors = get_dlrds_meta()

    if raw_ids_are_dataset_ids:
        mapped = np.full_like(label, fill_value=-1, dtype=np.int32)
        for dataset_id, contiguous_id in id_map.items():
            mapped[label == dataset_id] = contiguous_id
    else:
        mapped = label

    vis = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for idx, rgb in enumerate(stuff_colors):
        mask = mapped == idx
        # stuff_colors are RGB, but OpenCV image buffers are BGR.
        vis[mask] = np.array(rgb[::-1], dtype=np.uint8)
    return vis


def add_title_bar(image: np.ndarray, title: str, bar_h: int = 36) -> np.ndarray:
    h, w = image.shape[:2]
    bar = np.full((bar_h, w, 3), 245, dtype=np.uint8)
    cv2.putText(
        bar,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([bar, image])


def make_unique_id_panel(mask: np.ndarray, d2mask: np.ndarray, width: int) -> np.ndarray:
    mask_ids = np.unique(to_single_channel(mask))
    d2mask_ids = np.unique(to_single_channel(d2mask))

    lines = [
        f"masks unique ids ({len(mask_ids)}): {mask_ids.tolist()}",
        f"D2masks unique ids ({len(d2mask_ids)}): {d2mask_ids.tolist()}",
    ]

    line_h = 30
    pad_top = 16
    pad_bottom = 12
    panel_h = pad_top + pad_bottom + line_h * len(lines)
    panel = np.full((panel_h, width, 3), 250, dtype=np.uint8)

    for i, text in enumerate(lines):
        y = pad_top + (i + 1) * line_h - 8
        cv2.putText(
            panel,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
    return panel


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mask_path = os.path.join(repo_root, "datasets", "DLRSD", "masks", "Airplane00.png")
    d2mask_path = os.path.join(
        repo_root, "datasets", "DLRSD", "D2masks", "Airplane00.png"
    )
    out_dir = os.path.join(repo_root, "outputs")
    out_path = os.path.join(out_dir, "Airplane00_masks_vs_D2masks.png")

    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    d2mask = cv2.imread(d2mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read file: {mask_path}")
    if d2mask is None:
        raise FileNotFoundError(f"Cannot read file: {d2mask_path}")

    mask_gray = normalize_to_gray(mask)
    d2mask_gray = normalize_to_gray(d2mask)

    mask_vis = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    d2mask_vis = cv2.cvtColor(d2mask_gray, cv2.COLOR_GRAY2BGR)

    if mask_vis.shape[:2] != d2mask_vis.shape[:2]:
        d2mask_vis = cv2.resize(
            d2mask_vis, (mask_vis.shape[1], mask_vis.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    left = add_title_bar(mask_vis, "masks/Airplane00.png")
    right = add_title_bar(d2mask_vis, "D2masks/Airplane00.png")
    compare_gray = np.hstack([left, right])

    # Follow demo_visual_gt.py -> Visualizer.draw_sem_seg style:
    # masks are treated as dataset ids (1..17), D2masks as contiguous ids (0..16).
    mask_color = colorize_with_visualizer_logic(mask, raw_ids_are_dataset_ids=True)
    d2mask_color = colorize_with_visualizer_logic(d2mask, raw_ids_are_dataset_ids=False)
    if mask_color.shape[:2] != d2mask_color.shape[:2]:
        d2mask_color = cv2.resize(
            d2mask_color,
            (mask_color.shape[1], mask_color.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    left_color = add_title_bar(mask_color, "masks colorized (dataset id -> contiguous id)")
    right_color = add_title_bar(d2mask_color, "D2masks colorized (contiguous id)")
    compare_color = np.hstack([left_color, right_color])

    compare = np.vstack([compare_gray, compare_color])
    id_panel = make_unique_id_panel(mask, d2mask, compare.shape[1])
    compare = np.vstack([compare, id_panel])

    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out_path, compare)
    print(f"Saved comparison image to: {out_path}")

    # Optional preview window (press any key to close)
    # cv2.imshow("masks vs D2masks", compare)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
