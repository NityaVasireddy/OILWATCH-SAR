from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

from .preprocessing import preprocess


class OilSpillDataset(Dataset):

    def __init__(
        self,
        manifest,
        split,
        image_size=256,
        stride=256,
        positive_fraction=0.5,
    ):
        self.manifest = Path(manifest)
        self.split = split
        self.size = int(image_size)
        self.stride = int(stride)
        self.positive_fraction = float(positive_fraction)

        df = pd.read_csv(self.manifest)

        split_column = (
            "final_split"
            if "final_split" in df.columns
            else "split"
        )

        df = df[
            (df[split_column] == split)
            & (
                df["pair_valid"]
                .astype(str)
                .str.lower()
                == "true"
            )
        ].reset_index(drop=True)

        if df.empty:
            raise ValueError(
                f"No valid samples found for split='{split}'."
            )

        self.samples = []

        for _, row_data in df.iterrows():

            image_path = Path(row_data["image_path"])
            mask_path = Path(row_data["mask_path"])

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found: {mask_path}"
                )

            with rasterio.open(image_path) as src:
                height = src.height
                width = src.width

            positions = self._tile_positions(
                height,
                width,
                self.size,
                self.stride,
            )

            for row_start, col_start in positions:

                self.samples.append(
                    {
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "sample_id": row_data["sample_id"],
                        "row": row_start,
                        "col": col_start,
                    }
                )

        if not self.samples:
            raise ValueError(
                f"No tiles generated for split='{split}'."
            )

        self.positive_indices = []
        self.background_indices = []

        for index, item in enumerate(self.samples):

            with rasterio.open(item["mask_path"]) as src:

                mask = src.read(
                    1,
                    window=rasterio.windows.Window(
                        item["col"],
                        item["row"],
                        self.size,
                        self.size,
                    ),
                    boundless=True,
                    fill_value=0,
                )

            if np.any(mask > 0):
                self.positive_indices.append(index)
            else:
                self.background_indices.append(index)

        print(
            f"OilSpillDataset split={split}: "
            f"{len(df)} scenes -> {len(self.samples)} tiles"
        )

        print(
            f"  Positive tiles: {len(self.positive_indices)}"
        )

        print(
            f"  Background tiles: {len(self.background_indices)}"
        )

        self.training_indices = None

        if split == "train":

            if not self.positive_indices:
                raise ValueError(
                    "Training dataset contains no positive oil tiles."
                )

            if not self.background_indices:
                raise ValueError(
                    "Training dataset contains no background tiles."
                )

            total = len(self.samples)

            desired_positive = int(
                total * self.positive_fraction
            )

            desired_background = (
                total - desired_positive
            )

            rng = np.random.default_rng(42)

            positive_selected = rng.choice(
                self.positive_indices,
                size=desired_positive,
                replace=True,
            )

            background_selected = rng.choice(
                self.background_indices,
                size=desired_background,
                replace=True,
            )

            self.training_indices = np.concatenate(
                [
                    positive_selected,
                    background_selected,
                ]
            )

            rng.shuffle(self.training_indices)

            print(
                f"  Balanced training tiles: "
                f"{len(self.training_indices)}"
            )

            print(
                f"  Target positive fraction: "
                f"{self.positive_fraction:.2f}"
            )

    @staticmethod
    def _tile_positions(
        height,
        width,
        size,
        stride,
    ):

        rows = list(
            range(
                0,
                max(height - size + 1, 1),
                stride,
            )
        )

        cols = list(
            range(
                0,
                max(width - size + 1, 1),
                stride,
            )
        )

        last_row = max(height - size, 0)
        last_col = max(width - size, 0)

        if last_row not in rows:
            rows.append(last_row)

        if last_col not in cols:
            cols.append(last_col)

        positions = []

        for row in rows:
            for col in cols:
                positions.append(
                    (row, col)
                )

        return sorted(set(positions))

    def __len__(self):

        if self.training_indices is not None:
            return len(self.training_indices)

        return len(self.samples)

    def __getitem__(self, index):

        if self.training_indices is not None:
            real_index = int(
                self.training_indices[index]
            )
        else:
            real_index = index

        item = self.samples[real_index]

        image_path = item["image_path"]
        mask_path = item["mask_path"]

        row = item["row"]
        col = item["col"]

        with rasterio.open(image_path) as src:

            image = src.read(
                1,
                window=rasterio.windows.Window(
                    col,
                    row,
                    self.size,
                    self.size,
                ),
                boundless=True,
                fill_value=0,
            )

        with rasterio.open(mask_path) as src:

            mask = src.read(
                1,
                window=rasterio.windows.Window(
                    col,
                    row,
                    self.size,
                    self.size,
                ),
                boundless=True,
                fill_value=0,
            )

        image = np.asarray(
            image,
            dtype=np.float32,
        )

        mask = np.asarray(
            mask,
            dtype=np.uint8,
        )

        mask = (
            mask > 0
        ).astype(np.float32)

        x, _ = preprocess(
            image,
            size=self.size,
        )

        y = mask[None, ...]

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
        )