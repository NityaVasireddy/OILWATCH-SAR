from pathlib import Path
import hashlib
import pandas as pd
import rasterio

IMAGE_EXTENSIONS = {".tif", ".tiff"}


def raster_info(path):
    with rasterio.open(path) as src:
        return {
            "height": src.height,
            "width": src.width,
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": str(src.crs) if src.crs else "",
            "transform": str(src.transform),
        }


def collect_pairs(root):
    rows = []

    for split in ["train", "test"]:
        image_dir = root / split / "images"
        mask_dir = root / split / "masks"

        print(f"\nChecking {split}...")
        print(f"Images: {image_dir}")
        print(f"Masks : {mask_dir}")

        if not image_dir.exists():
            print(f"WARNING: image directory does not exist: {image_dir}")
            continue

        if not mask_dir.exists():
            print(f"WARNING: mask directory does not exist: {mask_dir}")
            continue

        images = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        masks = {
            p.name.lower(): p
            for p in mask_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        }

        print(f"Found {len(images)} images and {len(masks)} masks.")

        for image in images:
            mask = masks.get(image.name.lower())

            if mask is None:
                rows.append({
                    "sample_id": image.stem,
                    "scene_id": image.stem,
                    "split": split,
                    "image_path": str(image.resolve()),
                    "mask_path": "",
                    "pair_valid": False,
                    "reason": "missing_mask",
                })
                continue

            try:
                image_info = raster_info(image)
                mask_info = raster_info(mask)

                if image_info["height"] != mask_info["height"]:
                    valid = False
                    reason = "height_mismatch"

                elif image_info["width"] != mask_info["width"]:
                    valid = False
                    reason = "width_mismatch"

                elif image_info["count"] != mask_info["count"]:
                    valid = False
                    reason = "channel_mismatch"

                elif image_info["crs"] != mask_info["crs"]:
                    valid = False
                    reason = "crs_mismatch"

                elif image_info["transform"] != mask_info["transform"]:
                    valid = False
                    reason = "transform_mismatch"

                else:
                    valid = True
                    reason = "valid"

            except Exception as exc:
                valid = False
                reason = f"read_error:{type(exc).__name__}"

            rows.append({
                "sample_id": image.stem,
                "scene_id": image.stem,
                "split": split,
                "image_path": str(image.resolve()),
                "mask_path": str(mask.resolve()),
                "pair_valid": valid,
                "reason": reason,
            })

    return rows


def find_duplicates(root):
    files = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    hashes = {}

    for path in files:
        try:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes.setdefault(sha, []).append(str(path.resolve()))
        except Exception:
            pass

    duplicates = []

    for sha, paths in hashes.items():
        if len(paths) > 1:
            duplicates.append({
                "sha256": sha,
                "paths": "|".join(paths),
                "duplicate_count": len(paths),
            })

    return duplicates


def main():
    root = Path("dataset/raw")
    output_dir = Path("dataset/metadata")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise SystemExit(
            f"Dataset root does not exist: {root.resolve()}"
        )

    rows = collect_pairs(root)

    df = pd.DataFrame(rows)

    pairing_file = output_dir / "pairing_report.csv"
    df.to_csv(pairing_file, index=False)

    duplicates = find_duplicates(root)

    duplicate_file = output_dir / "duplicates.csv"

    pd.DataFrame(
        duplicates,
        columns=["sha256", "paths", "duplicate_count"],
    ).to_csv(duplicate_file, index=False)

    valid_count = (
        int(df["pair_valid"].sum())
        if not df.empty
        else 0
    )

    print("\n========================================")
    print("PAIRING COMPLETE")
    print("========================================")
    print(f"Total records : {len(df)}")
    print(f"Valid pairs   : {valid_count}")
    print(f"Duplicate groups: {len(duplicates)}")
    print(f"Report        : {pairing_file.resolve()}")
    print("========================================")

    if not df.empty:
        print("\nSummary:")
        print(
            df.groupby(["split", "reason"])
            .size()
            .to_string()
        )


if __name__ == "__main__":
    main()
