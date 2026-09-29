import datetime
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import OilSpillDataset
from .losses import combined_loss
from .metrics import binary_metrics
from .model import UNet


def main():

    # ============================================================
    # LOAD CONFIGURATION
    # ============================================================

    config_path = Path("configs/training.yaml")

    if not config_path.exists():
        raise FileNotFoundError(
            f"Training configuration not found: {config_path}"
        )

    cfg = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    # ============================================================
    # RANDOM SEEDS
    # ============================================================

    seed = int(cfg.get("seed", 42))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ============================================================
    # DEVICE
    # ============================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = device.type == "cuda"

    print()
    print("=" * 60)
    print("OILWATCH-SAR TRAINING")
    print("=" * 60)

    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())
    print("AMP:", use_amp)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("=" * 60)

    # ============================================================
    # DATASET SETTINGS
    # ============================================================

    manifest = "dataset/metadata/split_manifest.csv"

    image_size = int(
        cfg.get("image_size", 256)
    )

    batch_size = int(
        cfg.get("batch_size", 2)
    )

    epochs = int(
        cfg.get("epochs", 1)
    )

    num_workers = int(
        cfg.get("num_workers", 0)
    )

    learning_rate = float(
        cfg.get("learning_rate", 0.0001)
    )

    weight_decay = float(
        cfg.get("weight_decay", 0.00001)
    )

    threshold = float(
        cfg.get("threshold", 0.5)
    )

    patience = int(
        cfg.get("patience", 8)
    )

    # ============================================================
    # DATASETS
    # ============================================================

    print()
    print("Loading training dataset...")

    train_ds = OilSpillDataset(
        manifest,
        "train",
        image_size,
    )

    print()
    print("Loading validation dataset...")

    val_ds = OilSpillDataset(
        manifest,
        "val",
        image_size,
    )

    print()
    print("Training samples:", len(train_ds))
    print("Validation samples:", len(val_ds))

    if len(train_ds) == 0:
        raise RuntimeError(
            "Training dataset is empty."
        )

    if len(val_ds) == 0:
        raise RuntimeError(
            "Validation dataset is empty."
        )

    # ============================================================
    # DATALOADERS
    # ============================================================

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print()
    print("Batch size:", batch_size)
    print(
        "Training batches:",
        len(train_loader)
    )
    print(
        "Validation batches:",
        len(val_loader)
    )

    # ============================================================
    # MODEL
    # ============================================================

    print()
    print("Creating U-Net model...")

    model = UNet(
        in_channels=1,
        out_channels=1,
        base=16,
    ).to(device)

    print("Input channels: 1")
    print("Input size:", image_size)
    print("Architecture: U-Net")
    print("Base channels: 8")

    # ============================================================
    # OPTIMIZER
    # ============================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=2,
        factor=0.5,
    )

    # ============================================================
    # MIXED PRECISION
    # ============================================================

    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    # ============================================================
    # MODEL OUTPUT DIRECTORY
    # ============================================================

    model_dir = Path(
        "backend/model"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        model_dir / "best_model.pth"
    )

    metadata_path = (
        model_dir / "model_metadata.json"
    )

    # ============================================================
    # TRAINING VARIABLES
    # ============================================================

    best_dice = -1.0
    wait = 0

    training_history = []

    # ============================================================
    # TRAINING LOOP
    # ============================================================

    for epoch in range(epochs):

        print()
        print("=" * 60)
        print(
            f"EPOCH {epoch + 1}/{epochs}"
        )
        print("=" * 60)

        # --------------------------------------------------------
        # TRAIN
        # --------------------------------------------------------

        model.train()

        train_loss_total = 0.0

        train_batches = 0

        for batch_index, (x, y) in enumerate(
            train_loader,
            start=1,
        ):

            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # ----------------------------------------------------
            # FORWARD + LOSS
            # ----------------------------------------------------

            if use_amp:

                with torch.amp.autocast(
                    device_type="cuda"
                ):

                    logits = model(x)

                    loss = combined_loss(
                        logits,
                        y,
                    )

                scaler.scale(
                    loss
                ).backward()

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                logits = model(x)

                loss = combined_loss(
                    logits,
                    y,
                )

                loss.backward()

                optimizer.step()

            # ----------------------------------------------------
            # RECORD LOSS
            # ----------------------------------------------------

            loss_value = (
                loss.detach().item()
            )

            train_loss_total += (
                loss_value * len(x)
            )

            train_batches += 1

            # ----------------------------------------------------
            # PROGRESS
            # ----------------------------------------------------

            if (
                batch_index == 1
                or batch_index % 25 == 0
                or batch_index == len(train_loader)
            ):

                print(
                    f"Train batch "
                    f"{batch_index}/"
                    f"{len(train_loader)} "
                    f"| loss="
                    f"{loss_value:.5f}"
                )

        # --------------------------------------------------------
        # AVERAGE TRAIN LOSS
        # --------------------------------------------------------

        train_loss = (
            train_loss_total
            / len(train_ds)
        )

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

        print()
        print("Running validation...")

        model.eval()

        val_loss_total = 0.0

        probabilities = []
        ground_truth = []

        with torch.no_grad():

            for batch_index, (x, y) in enumerate(
                val_loader,
                start=1,
            ):

                x = x.to(
                    device,
                    non_blocking=True,
                )

                y = y.to(
                    device,
                    non_blocking=True,
                )

                # ------------------------------------------------
                # FORWARD
                # ------------------------------------------------

                if use_amp:

                    with torch.amp.autocast(
                        device_type="cuda"
                    ):

                        logits = model(x)

                        loss = combined_loss(
                            logits,
                            y,
                        )

                else:

                    logits = model(x)

                    loss = combined_loss(
                        logits,
                        y,
                    )

                # ------------------------------------------------
                # VALIDATION LOSS
                # ------------------------------------------------

                val_loss_total += (
                    loss.detach().item()
                    * len(x)
                )

                # ------------------------------------------------
                # PROBABILITY + GROUND TRUTH
                # ------------------------------------------------

                probabilities.append(
                    torch.sigmoid(
                        logits
                    )
                    .cpu()
                    .numpy()
                )

                ground_truth.append(
                    y.cpu().numpy()
                )

        # --------------------------------------------------------
        # AVERAGE VALIDATION LOSS
        # --------------------------------------------------------

        val_loss = (
            val_loss_total
            / len(val_ds)
        )

        probabilities = np.concatenate(
            probabilities,
            axis=0,
        )

        ground_truth = np.concatenate(
            ground_truth,
            axis=0,
        )

        # --------------------------------------------------------
        # METRICS
        # --------------------------------------------------------

        metrics = binary_metrics(
            probabilities,
            ground_truth,
            threshold,
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

        scheduler.step(
            dice
        )

        current_lr = optimizer.param_groups[0][
            "lr"
        ]

        # ========================================================
        # PRINT EPOCH RESULT
        # ========================================================

        print()
        print("-" * 60)

        print(
            f"Epoch {epoch + 1}/{epochs}"
        )

        print(
            f"Train Loss : {train_loss:.5f}"
        )

        print(
            f"Val Loss   : {val_loss:.5f}"
        )

        print(
            f"Dice       : {dice:.5f}"
        )

        print(
            f"IoU        : {iou:.5f}"
        )

        print(
            f"Precision  : {precision:.5f}"
        )

        print(
            f"Recall     : {recall:.5f}"
        )

        print(
            f"Learning Rate : {current_lr:.8f}"
        )

        print("-" * 60)

        # ========================================================
        # SAVE HISTORY
        # ========================================================

        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "learning_rate": current_lr,
            }
        )

        # ========================================================
        # SAVE BEST MODEL
        # ========================================================

        if dice > best_dice:

            best_dice = dice

            wait = 0

            checkpoint = {
                "state_dict": model.state_dict(),

                "architecture": "UNet",

                "input_channels": 1,

                "input_size": image_size,

                "base_channels": 16,

                "threshold": threshold,

                "best_val_dice": dice,

                "best_val_iou": iou,

                "best_val_precision": precision,

                "best_val_recall": recall,

                "trained_at": (
                    datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat()
                ),

                "preprocessing_version": "1.0",

                "device": str(device),
            }

            torch.save(
                checkpoint,
                checkpoint_path,
            )

            print()
            print(
                "NEW BEST MODEL SAVED"
            )

            print(
                "Path:",
                checkpoint_path,
            )

            print(
                "Best validation Dice:",
                best_dice,
            )

        else:

            wait += 1

            print()
            print(
                "Validation Dice did not improve."
            )

            print(
                f"Early stopping counter: "
                f"{wait}/{patience}"
            )

        # ========================================================
        # EARLY STOPPING
        # ========================================================

        if wait >= patience:

            print()
            print(
                "Early stopping triggered."
            )

            break

    # ============================================================
    # SAVE TRAINING METADATA
    # ============================================================

    metadata = {
        "architecture": "UNet",
        "input_channels": 1,
        "input_size": image_size,
        "base_channels": 16,
        "best_validation_dice": best_dice,
        "training_date": (
            datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
        ),
        "device": str(device),
        "mixed_precision": use_amp,
        "epochs_requested": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "threshold": threshold,
        "history": training_history,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # FINAL RESULT
    # ============================================================

    print()
    print("=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    print(
        "Best validation Dice:",
        best_dice,
    )

    print(
        "Model:",
        checkpoint_path,
    )

    print(
        "Metadata:",
        metadata_path,
    )

    if checkpoint_path.exists():

        print()
        print(
            "MODEL CHECKPOINT CREATED SUCCESSFULLY"
        )

    else:

        print()
        print(
            "WARNING: MODEL CHECKPOINT WAS NOT CREATED"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
