# ============================================
# LAB – ANALIZA STATYSTYCZNA: 4 OBRAZY
# koteczek, jelen, kosciol, krajobraz
# ============================================

import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.ndimage import generic_filter
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['figure.figsize'] = (10,6)
sns.set(style="whitegrid")


# ============================================
# --- FUNKCJE POMOCNICZE ---
# ============================================

def safe_imread_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Plik nie znaleziony: {path}")
    return img.astype(np.float32)


def compute_metrics(img):
    """Zwraca std, energy i korelacje kierunkowe."""
    img_f = img.astype(np.float64)
    std_val = float(np.std(img_f))
    energy = float(np.sum(img_f**2))

    def corr(a, b):
        if a.size < 2 or b.size < 2:
            return float('nan')
        return float(pearsonr(a, b)[0])

    Xh, Yh = img_f[:, :-1].ravel(), img_f[:, 1:].ravel()
    Xv, Yv = img_f[:-1, :].ravel(), img_f[1:, :].ravel()
    Xd1, Yd1 = img_f[:-1, :-1].ravel(), img_f[1:, 1:].ravel()
    Xd2, Yd2 = img_f[1:, :-1].ravel(), img_f[:-1, 1:].ravel()

    return {
        'std': std_val,
        'energy': energy,
        'corr_h': corr(Xh, Yh),
        'corr_v': corr(Xv, Yv),
        'corr_d1': corr(Xd1, Yd1),
        'corr_d2': corr(Xd2, Yd2)
    }


def plot_image_and_hist(img, title=""):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.hist(img.ravel(), bins=256, range=(0,255), color='gray')
    plt.title("Histogram intensywności")
    plt.xlabel("Intensywność")
    plt.ylabel("Liczność")

    plt.tight_layout()
    plt.show()


def plot_scatter_directions(img):
    pairs = [
        ("Pozioma", img[:, :-1].ravel(), img[:, 1:].ravel()),
        ("Pionowa", img[:-1, :].ravel(), img[1:, :].ravel()),
        ("Diag ↘", img[:-1, :-1].ravel(), img[1:, 1:].ravel()),
        ("Diag ↗", img[1:, :-1].ravel(), img[:-1, 1:].ravel()),
    ]

    plt.figure(figsize=(12,10))
    for i,(name,X,Y) in enumerate(pairs,1):
        plt.subplot(2,2,i)
        step = max(1, X.size // 20000)
        plt.scatter(X[::step], Y[::step], s=1, alpha=0.3)
        plt.title(name)
        plt.xlabel("I")
        plt.ylabel("I_sąsiad")
        plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


def local_variance(img, ksize=7):
    img_f = img.astype(np.float32)
    mean_local = cv2.blur(img_f, (ksize, ksize))
    mean_sq_local = cv2.blur(img_f**2, (ksize, ksize))
    var_local = mean_sq_local - mean_local**2
    var_local[var_local < 0] = 0.0
    return var_local


def plot_variance_2d_3d(var_map, downsample_step=6):
    plt.figure(figsize=(6,5))
    plt.imshow(var_map, cmap='inferno')
    plt.title("Mapa lokalnej wariancji (2D)")
    plt.axis('off')
    plt.colorbar()
    plt.show()

    H, W = var_map.shape
    step = max(1, min(downsample_step, max(1, min(H//100, W//100))))

    X = np.arange(0, W, step)
    Y = np.arange(0, H, step)
    Xg, Yg = np.meshgrid(X, Y)
    Z = var_map[::step, ::step]

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xg, Yg, Z, cmap='viridis', linewidth=0)
    ax.set_title("Mapa lokalnej wariancji — 3D")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Wariancja")
    plt.show()


# ============================================
# --- LISTA OBRAZÓW ---
# ============================================

image_paths = [
    "koteczek.jpg",
    "jelen.jpg",
    "kosciol.jpg",
    "krajobraz.jpg"
]

# katalog zapisowy
assets_dir = "report_assets"
os.makedirs(assets_dir, exist_ok=True)


# ============================================
# --- GŁÓWNA PĘTLA ANALIZY ---
# ============================================

results = {}

for path in image_paths:
    try:
        img = safe_imread_gray(path)
    except FileNotFoundError as e:
        print(e)
        continue

    metrics = compute_metrics(img)
    results[path] = {'metrics': metrics, 'shape': img.shape}

    print(f"\n=== {path} ===")
    print(f"Rozmiar: {img.shape[1]}x{img.shape[0]}")
    print(f"STD: {metrics['std']:.3f}, Energy: {metrics['energy']:.3e}")
    print(f"Korelacje: H={metrics['corr_h']:.4f}, V={metrics['corr_v']:.4f}, D↘={metrics['corr_d1']:.4f}, D↗={metrics['corr_d2']:.4f}")

    # wizualizacje
    plot_image_and_hist(img, title=path)
    plot_scatter_directions(img)

    # lokalna wariancja
    var_map = local_variance(img, ksize=7)
    plot_variance_2d_3d(var_map, downsample_step=6)

    # zapis miniaturek
    try:
        fn = os.path.splitext(os.path.basename(path))[0]
        plt.imsave(os.path.join(assets_dir, f"{fn}_img.png"), img, cmap='gray', vmin=0, vmax=255)
    except:
        print("Nie udało się zapisać miniatur dla", path)
