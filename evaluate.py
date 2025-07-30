import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import parameters
from utils.normalization import apply_normalization, unnormalize

# Dynamic model import
if parameters.MODEL_TYPE == "MLP":
    from model.mlp import MLP as ModelClass
elif parameters.MODEL_TYPE == "Transformer":
    from model.transformer import TransformerForwardModel as ModelClass
else:
    raise ValueError(f"Unsupported MODEL_TYPE: {parameters.MODEL_TYPE}")

# Set device
device = (
    torch.device("cpu")
    if parameters.USE_CPU_ONLY
    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
)


def evaluate_model(preds, truths):
    mse = ((preds - truths) ** 2).mean()
    mae = np.abs(preds - truths).mean()
    print(f"Evaluation Metrics:")
    print(f"🔹 MSE: {mse:.6f}")
    print(f"🔹 MAE: {mae:.6f}")


def load_data_and_model():
    # Load test data
    test_x = pd.read_csv(
        os.path.join(parameters.data_dir, "ADM", "testset", "test_g.csv"), header=None
    ).values.astype("float32")
    test_y = pd.read_csv(
        os.path.join(parameters.data_dir, "ADM", "testset", "test_s.csv"), header=None
    ).values.astype("float32")

    # Load normalization parameters
    x_max = np.load(os.path.join(parameters.output_dir, "normalization", "x_max.npy"))
    x_min = np.load(os.path.join(parameters.output_dir, "normalization", "x_min.npy"))
    y_max = np.load(os.path.join(parameters.output_dir, "normalization", "y_max.npy"))
    y_min = np.load(os.path.join(parameters.output_dir, "normalization", "y_min.npy"))

    # Normalize test data
    test_x_norm = apply_normalization(test_x, x_max, x_min)
    test_y_norm = apply_normalization(test_y, y_max, y_min)

    # Load model
    if parameters.MODEL_TYPE == "MLP":
        model = ModelClass(parameters.LINEAR, dropout=parameters.DROPOUT)
    elif parameters.MODEL_TYPE == "Transformer":
        model = ModelClass(
            input_dim=parameters.LINEAR[0],
            output_dim=parameters.LINEAR[-1],
            hidden_dim=parameters.HIDDEN_DIM,
            num_layers=parameters.TRANSFORMER_LAYERS,
            num_heads=parameters.TRANSFORMER_HEADS,
            dropout=parameters.DROPOUT_RATE,
        )
    else:
        raise ValueError("Invalid model type")

    model = model.to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(parameters.output_dir, parameters.DATA_SET, "best_model.pt"),
            map_location=device,
        )
    )
    model.eval()

    return model, test_x_norm, test_y_norm, test_x, test_y, y_max, y_min


def plot_spectrum_comparison(
    preds, truths, indices=[0, 10, 100, 500], save_dir="plots"
):
    os.makedirs(save_dir, exist_ok=True)
    for idx in indices:
        plt.figure()
        plt.plot(truths[idx], label="Ground Truth")
        plt.plot(preds[idx], label="Prediction")
        plt.title(f"Spectrum Comparison #{idx}")
        plt.xlabel("Frequency Index")          
        plt.ylabel("S-Parameter Value")      
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"spectrum_comparison_{idx}.png"))
        plt.close()


def plot_error_distributions(preds, truths, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    errors = np.abs(preds - truths)
    mse_per_sample = ((preds - truths) ** 2).mean(axis=1)

    # MAE distribution
    plt.figure()
    plt.hist(errors.mean(axis=1), bins=50)
    plt.title("MAE Distribution")
    plt.xlabel("Mean Absolute Error")         
    plt.ylabel("Number of Samples")         
    plt.savefig(os.path.join(save_dir, "mae_distribution.png"))
    plt.close()

    # MSE distribution
    plt.figure()
    plt.hist(mse_per_sample, bins=50)
    plt.title("MSE Distribution")
    plt.xlabel("Mean Squared Error")         
    plt.ylabel("Number of Samples")       
    plt.savefig(os.path.join(save_dir, "mse_distribution.png"))
    plt.close()


def main():
    model, x_norm, y_norm, x_raw, y_raw, y_max, y_min = load_data_and_model()

    with torch.no_grad():
        preds_norm = model(torch.from_numpy(x_norm).to(device)).cpu().numpy()

    preds = unnormalize(preds_norm, y_max, y_min)
    truths = y_raw

    results_dir = os.path.join(parameters.output_dir, parameters.DATA_SET)
    os.makedirs(results_dir, exist_ok=True)
    np.savetxt(os.path.join(results_dir, "test_y_pred.csv"), preds, delimiter=",")
    np.savetxt(os.path.join(results_dir, "test_y_true.csv"), truths, delimiter=",")
    np.savetxt(os.path.join(results_dir, "test_x.csv"), x_raw, delimiter=",")

    plot_spectrum_comparison(preds, truths, save_dir="plots")
    plot_error_distributions(preds, truths, save_dir="plots")

    evaluate_model(preds, truths)


if __name__ == "__main__":
    main()
