import math
import pandas as pd


def hav(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, h)))


def angle_diff(a, b):
    return abs((float(a) - float(b) + 180) % 360 - 180)


def score(distance_km, time_minutes, heading_diff, continuity=1.0):
    ds = max(0, 1 - min(float(distance_km) / 50, 1))
    ts = max(0, 1 - min(abs(float(time_minutes)) / 180, 1))
    hs = max(0, 1 - min(float(heading_diff) / 90, 1))
    cs = max(0, min(float(continuity), 1))
    total = 100 * (0.35 * ds + 0.25 * ts + 0.15 * hs + 0.25 * cs)
    return {
        "distance_score": 100 * ds,
        "temporal_score": 100 * ts,
        "heading_score": 100 * hs,
        "trajectory_score": 100 * ((ds + ts) / 2),
        "ais_continuity_score": 100 * cs,
        "correlation_score": total,
    }


def correlate_dataframe(df: pd.DataFrame, release_lat, release_lon, release_time=None):
    rows = []
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    release_ts = pd.to_datetime(release_time, utc=True, errors="coerce") if release_time else None
    for mmsi, g in d.groupby("MMSI", sort=False):
        g = g.dropna(subset=["latitude", "longitude", "timestamp"]).sort_values("timestamp")
        if g.empty:
            continue
        distances = g.apply(lambda r: hav(release_lat, release_lon, r.latitude, r.longitude), axis=1)
        idx = distances.idxmin()
        row = g.loc[idx]
        dist = float(distances.loc[idx])
        if release_ts is not None and not pd.isna(release_ts):
            dt = float(abs((row.timestamp - release_ts).total_seconds()) / 60.0)
        else:
            dt = 0.0
        heading = float(row.get("COG", 0.0)) if pd.notna(row.get("COG", float("nan"))) else 0.0
        continuity = min(1.0, len(g) / 20.0)
        metrics = score(dist, dt, 0.0, continuity)
        rows.append({
            "vessel_name": row.get("vessel_name") if "vessel_name" in row.index else None,
            "mmsi": str(mmsi),
            "vessel_type": row.get("vessel_type") if "vessel_type" in row.index else None,
            "nearest_distance_km": dist,
            "nearest_timestamp": row.timestamp.isoformat(),
            "time_difference_minutes": dt,
            "heading_difference_degrees": 0.0,
            "nearest_latitude": float(row.latitude),
            "nearest_longitude": float(row.longitude),
            "heading_at_nearest_position": heading,
            "ais_continuity": continuity,
            **metrics,
            "label": "Investigation Lead",
        })
    return sorted(rows, key=lambda x: x["correlation_score"], reverse=True)
