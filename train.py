import torch
import numpy as np
import pandas as pd
import os
from model import parameters
from model.forward import ForwardModel
from utils.data_loader import create_dataloaders
from utils.normalization import normalize, normalize_spectrum_columnwise
from sklearn.model_selection import train_test_split


def main():
    # === Load dataset ===
    data_x = pd.read_csv(
        f"{parameters.data_dir}/ADM/data_g.csv", header=None
    ).values.astype(np.float32)
    data_y = pd.read_csv(
        f"{parameters.data_dir}/ADM/data_s.csv", header=None
    ).values.astype(np.float32)

    print("\n✅ Data loaded:")
    print(f" 🔹 data_x shape: {data_x.shape}")
    print(f" 🔹 data_y shape: {data_y.shape}")

    # === Normalize inputs and outputs ===
    data_x, x_max, x_min = normalize(data_x)
    data_y, y_max, y_min = normalize_spectrum_columnwise(data_y)

    # === Save normalization parameters ===
    norm_dir = os.path.join(parameters.output_dir, "normalization")
    os.makedirs(norm_dir, exist_ok=True)
    np.save(os.path.join(norm_dir, "x_max.npy"), x_max)
    np.save(os.path.join(norm_dir, "x_min.npy"), x_min)
    np.save(os.path.join(norm_dir, "y_max.npy"), y_max)
    np.save(os.path.join(norm_dir, "y_min.npy"), y_min)
    print("✅ Normalization parameters saved.")

    # === Train-validation split ===
    train_x, val_x, train_y, val_y = train_test_split(
        data_x,
        data_y,
        test_size=parameters.TEST_RATIO,
        random_state=parameters.RAND_SEED,
    )
    print(f"✅ Data split:")
    print(f" 🔹 Training samples: {train_x.shape[0]}")
    print(f" 🔹 Validation samples: {val_x.shape[0]}")

    # === Create PyTorch DataLoaders ===
    train_loader, val_loader = create_dataloaders(
        train_x, train_y, val_x, val_y, parameters.BATCH_SIZE
    )
    print("✅ DataLoaders ready.")

    # === Create and train model ===
    model = ForwardModel(input_dim=data_x.shape[1], output_dim=data_y.shape[1])
    model.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
