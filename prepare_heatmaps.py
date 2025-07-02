import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def load_csvs(csv_dir):
    """Load and sort all CSV files in csv_dir."""
    csv_paths = sorted(Path(csv_dir).glob("*.csv"))
    dataframes = []
    for path in tqdm(csv_paths, desc="Parsing CSVs"):
        df = pd.read_csv(path)
        df = df.sort_values("timestamp")
        dataframes.append(df)
    return dataframes


def group_fixations(df):
    """Detect fixations for each image_id in df using a simple I-DT algorithm."""
    result = defaultdict(list)
    for image_id, dfi in df.groupby("image_id"):
        xs = dfi["x"].to_numpy()
        ys = dfi["y"].to_numpy()
        ts = dfi["timestamp"].to_numpy()
        n = len(dfi)
        start = 0
        while start < n:
            end = start
            while end < n and np.hypot(xs[end] - xs[start], ys[end] - ys[start]) <= 25:
                end += 1
            last = end - 1
            duration = ts[last] - ts[start]
            if duration >= 100:
                xc = xs[start:end].mean()
                yc = ys[start:end].mean()
                result[image_id].append((xc, yc, duration))
            start = end
    return result


def heatmap_from_fixations(fixations):
    """Create a normalized heatmap from a list of fixations."""
    arr = np.zeros((800, 800), dtype=np.float32)
    for x, y, dur in fixations:
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < 800 and 0 <= yi < 800:
            arr[yi, xi] += dur
    arr = gaussian_filter(arr, sigma=25)
    s = arr.sum()
    if s > 0:
        arr = (arr / s).astype(np.float32)
    return arr


def main():
    """Parse arguments and orchestrate heatmap creation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    datasets = load_csvs(args.csv_dir)
    per_image = defaultdict(lambda: np.zeros((800, 800), dtype=np.float32))
    for df in datasets:
        fix_by_img = group_fixations(df)
        for image_id, fixs in fix_by_img.items():
            per_image[image_id] += heatmap_from_fixations(fixs)

    image_files = [p for p in Path(args.img_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    count = 0
    for img_path in tqdm(sorted(image_files), desc="Saving heatmaps"):
        image_id = img_path.stem
        heatmap = per_image.get(image_id, np.zeros((800, 800), dtype=np.float32))
        s = heatmap.sum()
        if s > 0:
            heatmap = (heatmap / s).astype(np.float32)
        out_file = out_path / f"{image_id}_heatmap.npy"
        np.save(out_file, heatmap)
        count += 1

    print(f"Done. {count} heatmaps written to {args.out_dir}")


if __name__ == "__main__":
    main()
