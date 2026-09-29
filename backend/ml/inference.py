from pathlib import Path
import io
import time

import numpy as np
import torch
import rasterio
from rasterio.io import MemoryFile
from PIL import Image

from .model import UNet
from .preprocessing import preprocess


MODEL_PATH = Path("backend/model/best_model.pth")
TILE_SIZE = 256
STRIDE = 256


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "MODEL NOT LOADED - trained model checkpoint not found."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    input_channels = int(
        checkpoint.get("input_channels", 1)
    )

    base_channels = int(
        checkpoint.get("base_channels", 8)
    )

    model = UNet(
        in_channels=input_channels,
        out_channels=1,
        base=base_channels,
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    threshold = float(
        checkpoint.get("threshold", 0.5)
    )

    return model, threshold


def _predict_tile(model, tile):
    """
    Predict one SAR tile using exactly the preprocessing
    used during training.
    """

    tensor, preprocessing_info = preprocess(
        tile,
        size=TILE_SIZE,
    )

    x = torch.from_numpy(
        tensor
    ).unsqueeze(0).float()

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.sigmoid(logits)

    probability_map = (
        probabilities[0, 0]
        .cpu()
        .numpy()
    )

    return probability_map, preprocessing_info


def _predict_raster(model, threshold, src):
    """
    Run tiled inference on the original raster grid.

    The returned probability map and mask have exactly
    the same height and width as the source raster.
    """

    height = src.height
    width = src.width

    probability_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    prediction_count = np.zeros(
        (height, width),
        dtype=np.uint16,
    )

    preprocessing_examples = []

    for y in range(0, height, STRIDE):

        for x in range(0, width, STRIDE):

            tile_height = min(
                TILE_SIZE,
                height - y,
            )

            tile_width = min(
                TILE_SIZE,
                width - x,
            )

            tile = src.read(
                1,
                window=rasterio.windows.Window(
                    x,
                    y,
                    tile_width,
                    tile_height,
                ),
            )

            probability_tile, prep = _predict_tile(
                model,
                tile,
            )

            if len(preprocessing_examples) < 3:
                preprocessing_examples.append(prep)

            # Resize the model output back to the actual
            # edge-tile dimensions when necessary.
            if (
                probability_tile.shape[0] != tile_height
                or probability_tile.shape[1] != tile_width
            ):
                probability_image = Image.fromarray(
                    probability_tile.astype(np.float32),
                    mode="F",
                )

                probability_image = probability_image.resize(
                    (tile_width, tile_height),
                    Image.Resampling.BILINEAR,
                )

                probability_tile = np.asarray(
                    probability_image,
                    dtype=np.float32,
                )

            probability_sum[
                y:y + tile_height,
                x:x + tile_width,
            ] += probability_tile

            prediction_count[
                y:y + tile_height,
                x:x + tile_width,
            ] += 1

    probability_map = (
        probability_sum
        / np.maximum(prediction_count, 1)
    )

    mask = (
        probability_map >= threshold
    ).astype(np.uint8)

    return (
        probability_map,
        mask,
        preprocessing_examples,
    )


def _calculate_result(
    probability_map,
    mask,
    threshold,
):
    spill_pixels = int(mask.sum())
    total_pixels = int(mask.size)

    spill_fraction = (
        float(spill_pixels / total_pixels)
        if total_pixels > 0
        else 0.0
    )

    mean_probability = float(
        probability_map.mean()
    )

    max_probability = float(
        probability_map.max()
    )

    if spill_pixels > 0:
        spill_confidence = float(
            probability_map[mask == 1].mean()
        )
    else:
        spill_confidence = 0.0

    return {
        "mask": mask,
        "probability_map": probability_map,
        "spill_pixels": spill_pixels,
        "total_pixels": total_pixels,
        "spill_fraction": spill_fraction,
        "mean_probability": mean_probability,
        "max_probability": max_probability,
        "spill_confidence": spill_confidence,
        "threshold": float(threshold),
    }


def predict_tiff(
    image_path,
    output_path=None,
):
    """
    Run tiled U-Net inference on a real GeoTIFF.

    The prediction remains on the original raster grid,
    preserving the source CRS and geotransform.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"SAR image not found: {image_path}"
        )

    model, threshold = load_model()

    start = time.perf_counter()

    with rasterio.open(image_path) as src:

        probability_map, mask, prep_examples = (
            _predict_raster(
                model,
                threshold,
                src,
            )
        )

        profile = src.profile.copy()

        width = src.width
        height = src.height
        crs = src.crs
        transform = src.transform
        bounds = src.bounds
        resolution = src.res

    result = _calculate_result(
        probability_map,
        mask,
        threshold,
    )

    result.update(
        {
            "image": str(image_path),
            "image_width": width,
            "image_height": height,
            "crs": (
                str(crs)
                if crs is not None
                else None
            ),
            "transform": list(transform),
            "bounds": {
                "left": float(bounds.left),
                "bottom": float(bounds.bottom),
                "right": float(bounds.right),
                "top": float(bounds.top),
            },
            "resolution": [
                float(resolution[0]),
                float(resolution[1]),
            ],
            "preprocessing": {
                "tile_size": TILE_SIZE,
                "stride": STRIDE,
                "mode": "original_resolution_tiled",
                "examples": prep_examples,
            },
            "inference_time_seconds": (
                time.perf_counter() - start
            ),
        }
    )

    if output_path is not None:

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_profile = profile.copy()

        output_profile.update(
            {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 1,
                "dtype": "uint8",
                "compress": "lzw",
                "crs": crs,
                "transform": transform,
            }
        )

        with rasterio.open(
            output_path,
            "w",
            **output_profile,
        ) as dst:

            dst.write(
                mask,
                1,
            )

        result["mask_output"] = str(
            output_path
        )

    return result


class InferenceEngine:

    def __init__(self):
        self.model, self.threshold = load_model()

    def predict(self, image):

        start = time.perf_counter()

        # --------------------------------------------------
        # FastAPI UploadFile
        # --------------------------------------------------

        if hasattr(image, "file"):

            image_bytes = image.file.read()

            if not image_bytes:
                raise ValueError(
                    "Uploaded SAR image is empty."
                )

        elif isinstance(image, (bytes, bytearray)):

            image_bytes = bytes(image)

        else:

            # NumPy/image-array fallback
            image_array = np.asarray(
                image,
                dtype=np.float32,
            )

            tensor, prep = preprocess(
                image_array,
                size=TILE_SIZE,
            )

            x = torch.from_numpy(
                tensor
            ).unsqueeze(0).float()

            with torch.no_grad():

                logits = self.model(x)

                probabilities = torch.sigmoid(
                    logits
                )

            probability_map = (
                probabilities[0, 0]
                .cpu()
                .numpy()
            )

            mask = (
                probability_map >= self.threshold
            ).astype(np.uint8)

            result = _calculate_result(
                probability_map,
                mask,
                self.threshold,
            )

            return (
                result,
                prep,
                time.perf_counter() - start,
            )

        # --------------------------------------------------
        # Try to read as GeoTIFF
        # --------------------------------------------------

        try:

            with MemoryFile(
                image_bytes
            ) as memfile:

                with memfile.open() as src:

                    if src.count < 1:
                        raise ValueError(
                            "SAR raster contains no bands."
                        )

                    probability_map, mask, prep_examples = (
                        _predict_raster(
                            self.model,
                            self.threshold,
                            src,
                        )
                    )

                    result = _calculate_result(
                        probability_map,
                        mask,
                        self.threshold,
                    )

                    result.update(
                        {
                            "image_width": src.width,
                            "image_height": src.height,
                            "crs": (
                                str(src.crs)
                                if src.crs is not None
                                else None
                            ),
                            "transform": list(
                                src.transform
                            ),
                            "bounds": {
                                "left": float(
                                    src.bounds.left
                                ),
                                "bottom": float(
                                    src.bounds.bottom
                                ),
                                "right": float(
                                    src.bounds.right
                                ),
                                "top": float(
                                    src.bounds.top
                                ),
                            },
                            "resolution": [
                                float(src.res[0]),
                                float(src.res[1]),
                            ],
                            "preprocessing": {
                                "tile_size": TILE_SIZE,
                                "stride": STRIDE,
                                "mode": (
                                    "original_resolution_tiled"
                                ),
                                "examples": prep_examples,
                            },
                        }
                    )

                    elapsed = (
                        time.perf_counter()
                        - start
                    )

                    return (
                        result,
                        result["preprocessing"],
                        elapsed,
                    )

        except Exception as raster_error:

            # --------------------------------------------------
            # PNG/JPEG fallback
            # --------------------------------------------------

            try:

                pil_image = Image.open(
                    io.BytesIO(image_bytes)
                )

                image_array = np.asarray(
                    pil_image,
                    dtype=np.float32,
                )

            except Exception as image_error:

                raise ValueError(
                    "Unable to decode uploaded SAR image. "
                    f"Raster error: {raster_error}. "
                    f"Image error: {image_error}."
                ) from image_error

            tensor, prep = preprocess(
                image_array,
                size=TILE_SIZE,
            )

            x = torch.from_numpy(
                tensor
            ).unsqueeze(0).float()

            with torch.no_grad():

                logits = self.model(x)

                probabilities = torch.sigmoid(
                    logits
                )

            probability_map = (
                probabilities[0, 0]
                .cpu()
                .numpy()
            )

            mask = (
                probability_map >= self.threshold
            ).astype(np.uint8)

            result = _calculate_result(
                probability_map,
                mask,
                self.threshold,
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            return (
                result,
                prep,
                elapsed,
            )


def main():

    print("=" * 60)
    print("OILWATCH-SAR MODEL TEST")
    print("=" * 60)

    print()

    print(
        "Model checkpoint:",
        MODEL_PATH,
    )

    model, threshold = load_model()

    print(
        "MODEL LOADED SUCCESSFULLY"
    )

    print(
        "Threshold:",
        threshold,
    )

    print(
        "Architecture: U-Net"
    )

    print(
        "Input channels:",
        1,
    )

    print(
        "Base channels:",
        model.e1.net[0].out_channels,
    )

    print(
        "Inference mode:",
        "original-resolution tiled",
    )

    print()

    print(
        "The trained model is ready "
        "for real SAR inference."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()