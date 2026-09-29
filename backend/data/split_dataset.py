from pathlib import Path
import pandas as pd
import numpy as np


def main():
    pairing_file = Path("dataset/metadata/pairing_report.csv")
    output_file = Path("dataset/metadata/split_manifest.csv")

    if not pairing_file.exists():
        raise SystemExit(
            f"Pairing report not found: {pairing_file.resolve()}"
        )

    df = pd.read_csv(pairing_file)

    if df.empty:
        raise SystemExit("Pairing report is empty.")

    df = df[df["pair_valid"] == True].copy()

    if df.empty:
        raise SystemExit("No valid pairs found.")

    rng = np.random.default_rng(42)

    train_source = df[df["split"] == "train"].copy()
    test_source = df[df["split"] == "test"].copy()

    if train_source.empty:
        raise SystemExit("No training samples found.")

    if test_source.empty:
        raise SystemExit("No official test samples found.")

    # Shuffle only the original training samples.
    indices = np.arange(len(train_source))
    rng.shuffle(indices)

    train_source = train_source.iloc[indices].reset_index(drop=True)

    # Approximately 80/20 train/validation split.
    n_train = len(train_source)

    if n_train < 2:
        raise SystemExit(
            "Not enough training samples to create train/validation split."
        )

    n_val = max(1, round(0.20 * n_train))

    # Make sure at least one sample remains for training.
    n_val = min(n_val, n_train - 1)

    val_df = train_source.iloc[:n_val].copy()
    train_df = train_source.iloc[n_val:].copy()

    train_df["final_split"] = "train"
    val_df["final_split"] = "val"
    test_source["final_split"] = "test"

    manifest = pd.concat(
        [train_df, val_df, test_source],
        ignore_index=True
    )

    # Shuffle rows in the manifest only for presentation.
    manifest = manifest.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    manifest.to_csv(output_file, index=False)

    print("\n========================================")
    print("DATASET SPLIT COMPLETE")
    print("========================================")
    print(f"Total valid samples : {len(manifest)}")
    print(f"Training samples    : {(manifest.final_split == 'train').sum()}")
    print(f"Validation samples  : {(manifest.final_split == 'val').sum()}")
    print(f"Test samples        : {(manifest.final_split == 'test').sum()}")
    print(f"Manifest            : {output_file.resolve()}")
    print("========================================")

    print("\nSource folders preserved:")
    print(f"Original train samples: {len(train_source)}")
    print(f"Original test samples : {len(test_source)}")


if __name__ == "__main__":
    main()
