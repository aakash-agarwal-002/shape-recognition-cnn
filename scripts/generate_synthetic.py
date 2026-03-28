import numpy as np
import json
import os
from pathlib import Path

# Paths relative to the project root
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "raw" / "synthetic"
DATA_FILE = OUTPUT_DIR / "synthetic_strokes.json"
LABEL_FILE = BASE_DIR / "data" / "processed" / "labels.json" # Keep a copy for reference

N = 200
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHAPE_NAMES = [
    "ellipse",
    "line",
    "triangle",
    "rectangle",
    "pentagon",
    "hexagon",
    "star",
    "zigzag",
    "arc",
    "heart",
    "diamond",
    "arrow",
    "double_arrow",
    "cloud",
    "message",
    "parallelogram",
]
POLYGONAL_SHAPES = {"triangle", "rectangle", "pentagon", "hexagon", "diamond", "parallelogram"}
OPEN_SHAPES = {"line", "zigzag", "arc", "arrow", "double_arrow"}
DIRECTIONAL_SHAPES = {"arrow", "double_arrow", "message"}

# ======================
# UTILS
# ======================

def to_f(x):
    return np.array(x, dtype=np.float32)

def resample(pts, n=120):
    d = np.sqrt(((np.diff(pts, axis=0))**2).sum(axis=1))
    d = np.insert(d, 0, 0)
    t = np.cumsum(d)
    if t[-1] == 0:
        return pts
    t /= t[-1]
    new_t = np.linspace(0,1,n)

    out = []
    for i in range(2):
        out.append(np.interp(new_t, t, pts[:,i]))
    return np.stack(out, axis=1)

def smooth(pts, k=4):
    for _ in range(k):
        pts = (np.roll(pts,1,axis=0)+pts+np.roll(pts,-1,axis=0))/3
    return pts

def wobble(pts, scale=0.02):
    n = len(pts)
    noise = np.random.normal(0, scale, n)
    for _ in range(5):
        noise = (np.roll(noise,1)+noise+np.roll(noise,-1))/3
    pts[:,0] += noise
    pts[:,1] += np.roll(noise, 5)
    return pts

# ======================
# WIDTH (single stroke)
# ======================

def add_width(pts):
    if np.random.rand() < 0.5:
        return pts

    d = np.gradient(pts, axis=0)
    norm = np.stack([-d[:,1], d[:,0]], axis=1)
    norm /= (np.linalg.norm(norm, axis=1, keepdims=True) + 1e-8)

    w = np.random.uniform(0.01, 0.04)
    w_profile = w * (1 + 0.3*np.sin(np.linspace(0,2*np.pi,len(pts))))
    w_profile = np.clip(w_profile, 0, 0.01)

    p1 = pts + norm * w_profile[:,None]
    p2 = pts - norm * w_profile[:,None]

    return np.vstack([p1, p2[::-1]])

# ======================
# OVERLAP (extension)
# ======================

def extend_stroke(pts, name):
    if name in {"line","arrow","double_arrow","arc"}:
        return pts

    if np.random.rand() < 0.5:
        return pts

    d = pts[-1] - pts[-2]
    d /= (np.linalg.norm(d)+1e-8)

    extra_len = np.random.randint(10,30)
    ext = [pts[-1]]

    for i in range(extra_len):
        step = d + np.random.normal(0,0.02,2)
        ext.append(ext[-1] + step)

    return np.vstack([pts, np.array(ext)])

# ======================
# OPENING (controlled)
# ======================

def opening(pts, name):
    if name in {"line","zigzag","arc","arrow","double_arrow"} | POLYGONAL_SHAPES:
        return pts

    n = len(pts)

    # smaller, controlled gap so identity is preserved
    cut = int(n * np.random.uniform(0.05, 0.12))
    cut = min(cut, n - 10)

    start = np.random.randint(0, n - cut)

    # create a gap using NaNs (breaks the stroke visually)
    gap = np.full((cut, 2), np.nan, dtype=np.float32)

    return np.concatenate([pts[:start], gap, pts[start+cut:]])

# ======================
# TRANSFORM
# ======================

def transform(
    pts,
    rotate=True,
    rotation_deg=180,
    scale_range=(0.5, 1.5),
    translate_range=(-0.5, 0.5),
):
    pts *= np.random.uniform(scale_range[0], scale_range[1])

    if rotate and rotation_deg > 0:
        t = np.deg2rad(np.random.uniform(-rotation_deg, rotation_deg))
        R = np.array([[np.cos(t),-np.sin(t)],
                      [np.sin(t), np.cos(t)]])
        pts = pts @ R.T

    pts += np.random.uniform(translate_range[0], translate_range[1], 2)
    return pts


def get_generation_profile(name):
    profile = {
        "wobble": 0.01,
        "rotate": True,
        "rotation_deg": 180,
        "scale_range": (0.75, 1.25),
        "translate_range": (-0.35, 0.35),
    }

    if name in SMOOTH:
        profile["wobble"] = 0.02
    elif name == "line":
        profile.update(
            {
                "wobble": 0.0015,
                "rotate": False,
                "scale_range": (0.9, 1.1),
                "translate_range": (-0.2, 0.2),
            }
        )
    elif name in {"arc", "zigzag"}:
        profile.update(
            {
                "wobble": 0.004,
                "rotation_deg": 20,
                "scale_range": (0.9, 1.1),
                "translate_range": (-0.25, 0.25),
            }
        )
    elif name in POLYGONAL_SHAPES:
        profile.update(
            {
                "wobble": 0.003,
                "rotation_deg": 20 if name != "diamond" else 0,
                "scale_range": (0.92, 1.08),
                "translate_range": (-0.22, 0.22),
            }
        )
    elif name in DIRECTIONAL_SHAPES:
        profile.update(
            {
                "wobble": 0.003,
                "rotation_deg": 15,
                "scale_range": (0.95, 1.08),
                "translate_range": (-0.22, 0.22),
            }
        )

    return profile

# ======================
# SHAPES
# ======================

def circle():
    t = np.linspace(0,2*np.pi,200)
    return to_f(np.stack([np.cos(t), np.sin(t)], axis=1))

def ellipse():
    t = np.linspace(0,2*np.pi,200)
    return to_f(np.stack([1.6*np.cos(t), np.sin(t)], axis=1))

def arc():
    t = np.linspace(0,np.pi,150)
    return to_f(np.stack([np.cos(t), np.sin(t)], axis=1))

def line():
    t = np.linspace(0,1,100)
    return to_f(np.stack([t, np.zeros_like(t)], axis=1))

def polygon(n):
    ang = np.linspace(0,2*np.pi,n,endpoint=False)
    r = 1 + np.random.uniform(-0.2,0.2,n)
    pts = np.stack([r*np.cos(ang), r*np.sin(ang)], axis=1)
    return to_f(np.vstack([pts, pts[0]]))

def triangle(): return polygon(3)
def pentagon(): return polygon(5)
def hexagon(): return polygon(6)

def square():
    pts = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]], dtype=np.float32)
    pts += np.random.normal(0,0.08,pts.shape)
    return pts

def rectangle():
    w = np.random.uniform(1,2)
    h = np.random.uniform(0.5,1.2)
    pts = np.array([[0,0],[w,0],[w,h],[0,h],[0,0]], dtype=np.float32)
    pts += np.random.normal(0,0.08,pts.shape)
    return pts

def zigzag():
    waves = np.random.randint(3,7)
    x = np.linspace(0,1,120)
    y = 0.3*np.sin(waves*np.pi*x)
    return to_f(np.stack([x,y], axis=1))

# ===== STAR (two modes) =====
def star():
    if np.random.rand() < 0.5:
        t = np.linspace(0,2*np.pi,10,endpoint=False)
        r = np.where(np.arange(10)%2==0,1,0.4)
        pts = np.stack([r*np.cos(t), r*np.sin(t)], axis=1)
        return to_f(np.vstack([pts, pts[0]]))
    else:
        return to_f([
            [0,1],[0.3,0],[1,0.3],
            [0.3,-0.2],[0.6,-1],
            [0,-0.4],[-0.6,-1],
            [-0.3,-0.2],[-1,0.3],
            [-0.3,0]
        ])

def diamond():
    return to_f([[0,1],[1,0],[0,-1],[-1,0],[0,1]])

def heart():
    t = np.linspace(0,2*np.pi,200)
    x = 16*np.sin(t)**3
    y = 13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t)
    return to_f(np.stack([x,y], axis=1)/20)

# ===== ARROWS (FINAL CLEAN) =====

def arrow():
    L = 1.0
    h = np.random.uniform(0.2,0.35)

    return to_f([
        [0,0],[L,0],
        [L,0],[L-h,h],
        [L,0],[L-h,-h]
    ])

def double_arrow():
    L = 1.5
    h = np.random.uniform(0.2, 0.4)

    return to_f([
        [-L, 0],
        [-L + h,  h],
        [-L, 0],
        [-L + h, -h],
        [-L, 0],
        [L, 0],
        [L - h,  h],
        [L, 0],
        [L - h, -h]
    ])

def cloud():
    t = np.linspace(0,2*np.pi,200)
    x = 1.6*np.cos(t)
    y = 0.9*np.sin(t)
    bumps = 0.2*np.sin(5*t)
    return to_f(np.stack([x*(1+bumps), y*(1+bumps)], axis=1))

def message():
    return to_f([
        [-1,0.5],[1,0.5],[1,-0.5],[0,-0.5],
        [-0.3,-1],[-0.3,-0.5],[-1,-0.5],[-1,0.5]
    ])

def parallelogram():
    return to_f([[0,0],[1,0],[1.5,1],[0.5,1],[0,0]])

# ======================
# PIPELINE
# ======================

SMOOTH = {"circle","ellipse","cloud","heart"}

SHAPES = [
    ("ellipse", circle),
    ("line", line),
    ("triangle", triangle),
    ("rectangle", square),
    ("rectangle", rectangle),
    ("ellipse", ellipse),
    ("pentagon", pentagon),
    ("hexagon", hexagon),
    ("star", star),
    ("zigzag", zigzag),
    ("arc", arc),
    ("heart", heart),
    ("diamond", diamond),
    ("arrow", arrow),
    ("double_arrow", double_arrow),
    ("cloud", cloud),
    ("message", message),
    ("parallelogram", parallelogram),
]

def generate_synthetic_data(num_samples=200, output_path=DATA_FILE):
    data = []
    label_to_idx = {name: i for i, name in enumerate(SHAPE_NAMES)}

    for name, fn in SHAPES:
        label = label_to_idx[name]
        print(f"Generating {num_samples} samples for {name}...")

        for _ in range(num_samples):
            pts = fn()
            profile = get_generation_profile(name)

            if name in SMOOTH:
                pts = resample(pts, 120)
                pts = smooth(pts, 3)
            else:
                pts = resample(pts, 80)

            pts = wobble(pts, profile["wobble"])
            pts = opening(pts, name)
            pts = transform(
                pts,
                rotate=profile["rotate"],
                rotation_deg=profile["rotation_deg"],
                scale_range=profile["scale_range"],
                translate_range=profile["translate_range"],
            )

            if len(pts) < 10:
                continue

            data.append({
                "points": pts.tolist(),
                "label": label
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)
    
    print(f"DONE: {len(data)} synthetic samples saved to {output_path}")

if __name__ == "__main__":
    generate_synthetic_data(num_samples=N)
    print(f"Saved synthetic strokes to: {DATA_FILE}")
