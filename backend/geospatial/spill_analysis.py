import numpy as np
from scipy import ndimage


def analyze(mask, prob=None, resolution=None):
    m = np.asarray(mask).astype(bool)
    n = int(m.sum())
    total = m.size
    if n == 0:
        return {"detected_pixel_count": 0, "coverage_percent": 0.0, "message": "No spill predicted."}

    ys, xs = np.where(m)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    cy, cx = float(ys.mean()), float(xs.mean())
    perimeter = float((m ^ ndimage.binary_erosion(m)).sum())
    pts = np.column_stack([xs - cx, ys - cy])
    cov = np.cov(pts, rowvar=False) if len(pts) > 1 else np.eye(2)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vec = vecs[:, 0]
    major = float(4 * np.sqrt(max(vals[0], 0)))
    minor = float(4 * np.sqrt(max(vals[1], 0)))
    orient = float(np.degrees(np.arctan2(vec[1], vec[0])))

    out = {
        "detected_pixel_count": n,
        "coverage_percent": 100 * n / total,
        "bounding_box_pixels": {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1},
        "centroid_pixel": {"x": cx, "y": cy},
        "perimeter_pixels": perimeter,
        "major_axis_pixels": major,
        "minor_axis_pixels": minor,
        "orientation_degrees": orient,
        "physical_area": None,
        "area_note": "Physical area unavailable; showing pixel area.",
    }
    if resolution and len(resolution) == 2:
        pixel_x, pixel_y = abs(float(resolution[0])), abs(float(resolution[1]))
        if pixel_x > 0 and pixel_y > 0:
            out["physical_area"] = n * pixel_x * pixel_y
            out["physical_area_units"] = "square CRS units"
            out["area_note"] = "Physical area calculated from raster pixel resolution."

    if prob is not None:
        vals = np.asarray(prob)[m]
        out["model_confidence"] = {
            "min_probability": float(vals.min()),
            "max_probability": float(vals.max()),
            "mean_probability": float(vals.mean()),
            "median_probability": float(np.median(vals)),
        }
    return out
