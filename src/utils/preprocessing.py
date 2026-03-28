import numpy as np
import cv2

def stroke_to_image(points, size=64, line_width=2, padding_ratio=0.08):
    pts = np.array(points, dtype=np.float32)

    # remove NaNs
    pts = pts[~np.isnan(pts).any(axis=1)]

    if len(pts) < 2:
        return np.zeros((size, size), dtype=np.uint8)

    min_vals = pts.min(axis=0)
    max_vals = pts.max(axis=0)
    range_vals = max_vals - min_vals
    max_range = float(np.max(range_vals))
    if max_range == 0:
        max_range = 1.0

    # Preserve aspect ratio so a nearly-flat line does not get stretched into
    # a tall zigzag by per-axis normalization.
    offset = (max_range - range_vals) / 2.0
    pts = (pts - min_vals + offset) / max_range
    pts = np.clip(pts, 0.0, 1.0)

    # Keep a border around the stroke so the rasterized shape does not get
    # clipped against the image edges after normalization.
    if padding_ratio > 0:
        inner_scale = max(1.0 - 2.0 * padding_ratio, 1e-6)
        pts = pts * inner_scale + padding_ratio

    img = np.zeros((size, size), dtype=np.uint8)

    for i in range(len(pts)-1):
        x1, y1 = np.clip((pts[i] * (size-1)).astype(int), 0, size - 1)
        x2, y2 = np.clip((pts[i+1] * (size-1)).astype(int), 0, size - 1)
        cv2.line(img, (x1, y1), (x2, y2), 255, line_width)

    return img

def points_to_model_input(points, size=64, line_width=2, padding_ratio=0.08):
    img = stroke_to_image(
        points,
        size=size,
        line_width=line_width,
        padding_ratio=padding_ratio,
    ).astype(np.float32) / 255.0
    img = 1.0 - img
    return np.expand_dims(img, 0)
