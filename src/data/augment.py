import numpy as np

OPEN_SHAPES = {"line", "zigzag", "arc", "arrow", "double_arrow"}
POLYGONAL_SHAPES = {"triangle", "rectangle", "pentagon", "hexagon", "diamond", "parallelogram"}
DIRECTIONAL_SHAPES = {"arrow", "double_arrow", "message"}


def get_augmentation_profile(label):
    profile = {
        "rotate": True,
        "rotation_deg": 45,
        "flip_x_prob": 0.2,
        "flip_y_prob": 0.2,
        "scale_x": (0.85, 1.2),
        "scale_y": (0.85, 1.2),
        "shear": (-0.1, 0.1),
        "drop_prob": 0.0,
        "jitter_mul": 0.01,
        "jitter_min": 0.005,
        "translate_mul": 0.03,
        "translate_min": 0.01,
    }

    if label in {"diamond", "line"}:
        profile["rotate"] = False

    if label in {"line", "arc"}:
        profile.update(
            {
                "flip_x_prob": 0.0,
                "flip_y_prob": 0.0,
                "scale_x": (0.98, 1.04),
                "scale_y": (0.98, 1.04),
                "shear": (-0.005, 0.005),
                "jitter_mul": 0.0015,
                "jitter_min": 0.0008,
                "translate_mul": 0.008,
                "translate_min": 0.003,
            }
        )
    elif label in POLYGONAL_SHAPES:
        profile.update(
            {
                "rotation_deg": 20,
                "flip_x_prob": 0.1,
                "flip_y_prob": 0.1,
                "scale_x": (0.97, 1.04),
                "scale_y": (0.97, 1.04),
                "shear": (-0.015, 0.015),
                "jitter_mul": 0.0025,
                "jitter_min": 0.0015,
                "translate_mul": 0.012,
                "translate_min": 0.004,
            }
        )
    elif label in DIRECTIONAL_SHAPES:
        profile.update(
            {
                "flip_x_prob": 0.0,
                "flip_y_prob": 0.0,
                "rotation_deg": 15,
                "scale_x": (0.95, 1.06),
                "scale_y": (0.95, 1.06),
                "shear": (-0.01, 0.01),
                "jitter_mul": 0.003,
                "jitter_min": 0.0015,
                "translate_mul": 0.012,
                "translate_min": 0.004,
            }
        )

    return profile


def augment_points(points, source, label=None):
    pts = np.array(points, dtype=np.float32)
    pts = pts[~np.isnan(pts).any(axis=1)]

    if len(pts) < 2:
        return points

    if np.random.rand() < 0.5:
        pts = pts[::-1].copy()

    min_vals = pts.min(axis=0)
    max_vals = pts.max(axis=0)
    diag = np.linalg.norm(max_vals - min_vals) + 1e-6

    is_closed_like = (
        label not in OPEN_SHAPES
        and np.linalg.norm(pts[0] - pts[-1]) < 0.2 * diag
    )

    if is_closed_like and np.random.rand() < 0.7:
        shift = np.random.randint(0, len(pts))
        pts = np.roll(pts, shift, axis=0)

    center = pts.mean(axis=0, keepdims=True)
    pts = pts - center

    profile = get_augmentation_profile(label)

    if profile["rotate"]:
        angle = np.deg2rad(np.random.uniform(-profile["rotation_deg"], profile["rotation_deg"]))
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        pts = pts @ rotation.T

    if np.random.rand() < profile["flip_x_prob"]:
        pts[:, 0] *= -1
    if np.random.rand() < profile["flip_y_prob"]:
        pts[:, 1] *= -1

    scale_x = np.random.uniform(*profile["scale_x"])
    scale_y = np.random.uniform(*profile["scale_y"])
    pts[:, 0] *= scale_x
    pts[:, 1] *= scale_y

    shear = np.random.uniform(*profile["shear"])
    shear_matrix = np.array([[1.0, shear], [0.0, 1.0]], dtype=np.float32)
    pts = pts @ shear_matrix.T

    if profile["drop_prob"] > 0 and np.random.rand() < profile["drop_prob"] and len(pts) > 24:
        keep = np.ones(len(pts), dtype=bool)
        drop_idx = np.random.choice(
            np.arange(1, len(pts) - 1),
            size=1,
            replace=False,
        )
        keep[drop_idx] = False
        pts = pts[keep]

    jitter_scale = max(diag * profile["jitter_mul"], profile["jitter_min"])
    pts += np.random.normal(0, jitter_scale, pts.shape).astype(np.float32)

    translation_scale = max(diag * profile["translate_mul"], profile["translate_min"])
    pts += np.random.normal(0, translation_scale, (1, 2)).astype(np.float32)

    if source == "browser":
        pts += np.random.normal(0, jitter_scale * 0.25, pts.shape).astype(np.float32)

    pts += center

    return pts.tolist()
