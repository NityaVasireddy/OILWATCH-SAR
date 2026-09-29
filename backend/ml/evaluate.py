from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import OilSpillDataset
from .metrics import binary_metrics
from .model import UNet


MODEL_PATH = Path(
    "backend/model/best_model.pth"
)

MANIFEST_PATH = (
    "dataset/metadata/split_manifest.csv"
)

CONFIG_PATH = (
    "configs/training.yaml"
)


def load_checkpoint():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "MODEL NOT LOADED — "
            "backend/model/best_model.pth "
            "was not found."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    return checkpoint


def main():

    print()
    print("=" * 60)
    print("OILWATCH-SAR TEST SET EVALUATION")
    print("=" * 60)

    # ------------------------------------------------------------
    # LOAD CONFIG
    # ------------------------------------------------------------

    if not Path(CONFIG_PATH).exists():

        raise FileNotFoundError(
            f"Configuration not found: {CONFIG_PATH}"
        )

    cfg = yaml.safe_load(
        Path(CONFIG_PATH).read_text(
            encoding="utf-8"
        )
    )

    image_size = int(
        cfg.get(
            "image_size",
            256,
        )
    )

    batch_size = int(
        cfg.get(
            "batch_size",
            2,
        )
    )

    threshold = float(
        cfg.get(
            "threshold",
            0.5,
        )
    )

    # ------------------------------------------------------------
    # LOAD CHECKPOINT
    # ------------------------------------------------------------

    print()
    print("Loading trained model...")

    checkpoint = load_checkpoint()

    input_channels = int(
        checkpoint.get(
            "input_channels",
            1,
        )
    )

    base_channels = int(
        checkpoint.get(
            "base_channels",
            8,
        )
    )

    checkpoint_threshold = float(
        checkpoint.get(
            "threshold",
            threshold,
        )
    )

    print(
        "Architecture:",
        checkpoint.get(
            "architecture",
            "UNet",
        )
    )

    print(
        "Input channels:",
        input_channels,
    )

    print(
        "Base channels:",
        base_channels,
    )

    print(
        "Checkpoint validation Dice:",
        checkpoint.get(
            "best_val_dice",
            "N/A",
        )
    )

    print(
        "Inference threshold:",
        checkpoint_threshold,
    )

    # ------------------------------------------------------------
    # DEVICE
    # ------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    # ------------------------------------------------------------
    # CREATE MODEL USING CHECKPOINT ARCHITECTURE
    # ------------------------------------------------------------

    model = UNet(
        in_channels=input_channels,
        out_channels=1,
        base=base_channels,
    ).to(device)

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.eval()

    print()
    print(
        "MODEL LOADED SUCCESSFULLY"
    )

    # ------------------------------------------------------------
    # TEST DATASET
    # ------------------------------------------------------------

    print()
    print(
        "Loading official test dataset..."
    )

    test_ds = OilSpillDataset(
        MANIFEST_PATH,
        "test",
        image_size,
    )

    if len(test_ds) == 0:

        raise RuntimeError(
            "No test samples were found."
        )

    print(
        "Test samples:",
        len(test_ds),
    )

    # ------------------------------------------------------------
    # TEST DATALOADER
    # ------------------------------------------------------------

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(
        "Test batches:",
        len(test_loader),
    )

    # ------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------

    probabilities = []
    ground_truth = []

    print()
    print(
        "Running inference on test set..."
    )

    with torch.no_grad():

        for batch_index, (x, y) in enumerate(
            test_loader,
            start=1,
        ):

            x = x.to(device)

            logits = model(x)

            probs = torch.sigmoid(
                logits
            )

            probabilities.append(
                probs.cpu().numpy()
            )

            ground_truth.append(
                y.cpu().numpy()
            )

            if (
                batch_index == 1
                or batch_index % 25 == 0
                or batch_index == len(test_loader)
            ):

                print(
                    f"Test batch "
                    f"{batch_index}/"
                    f"{len(test_loader)}"
                )

    # ------------------------------------------------------------
    # COMBINE RESULTS
    # ------------------------------------------------------------

    probabilities = np.concatenate(
        probabilities,
        axis=0,
    )

    ground_truth = np.concatenate(
        ground_truth,
        axis=0,
    )

    # ------------------------------------------------------------
    # CALCULATE METRICS
    # ------------------------------------------------------------

    metrics = binary_metrics(
        probabilities,
        ground_truth,
        checkpoint_threshold,
    )

    dice = float(
        metrics.get(
            "dice",
            0.0,
        )
    )

    iou = float(
        metrics.get(
            "iou",
            0.0,
        )
    )

    precision = float(
        metrics.get(
            "precision",
            0.0,
        )
    )

    recall = float(
        metrics.get(
            "recall",
            0.0,
        )
    )

    # ------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------

    print()
    print("=" * 60)
    print("OFFICIAL TEST SET RESULTS")
    print("=" * 60)

    print(
        f"Dice      : {dice:.6f}"
    )

    print(
        f"IoU       : {iou:.6f}"
    )

    print(
        f"Precision : {precision:.6f}"
    )

    print(
        f"Recall    : {recall:.6f}"
    )

    print(
        f"Threshold : {checkpoint_threshold:.3f}"
    )

    print(
        f"Test tiles: {len(test_ds)}"
    )

    print("=" * 60)

    # ------------------------------------------------------------
    # INTERPRETATION
    # ------------------------------------------------------------

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "These metrics are calculated on the "
        "held-out test split."
    )

    print(
        "The current model was trained for only "
        "1 epoch with base_channels=8, so these "
        "results are an initial baseline, not "
        "a final production-quality model."
    )

    print("=" * 60)


if __name__ == "__main__":

    main()