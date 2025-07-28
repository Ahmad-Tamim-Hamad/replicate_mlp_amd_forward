"""
This file serves as an evaluation interface for the network
"""

# Built-in
import os
import sys

# Optional: adjust this to your actual project root if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Torch / NumPy
import numpy as np

# Own modules
import flag_reader
from class_wrapper import Network
from model_maker import Forward
from utils import data_reader
from utils.evaluation_helper import plotMSELossDistrib


def evaluate_from_model(model_dir):
    """
    Evaluation interface:
    1. Retrieve the flags
    2. Load the dataset
    3. Initialize the network
    4. Run evaluation and plot MSE distribution
    """
    if model_dir.startswith("models"):
        model_dir = model_dir[7:]  # Remove "models/" prefix if present
        print("after removing prefix models/, now model_dir is:", model_dir)

    print("Retrieving flag object for parameters")
    print(model_dir)

    # Load flags from the saved model directory
    flags = flag_reader.load_flags(os.path.join("models", model_dir))
    flags.eval_model = model_dir
    flags.skip_connection = True
    flags.test_ratio = 0.2 

    # Load data using flags
    train_loader, test_loader = data_reader.read_data(flags)

    print("Making network now")
    dim_g = test_loader.dataset[0][0].shape[0]
    dim_s = test_loader.dataset[0][1].shape[0]

    ntwk = Network(dim_g=dim_g, dim_s=dim_s, inference_mode=True, saved_model=flags.eval_model)

    print("Start eval now:")
    test_x = test_loader.dataset[:][0]
    test_y = test_loader.dataset[:][1]

    # Run evaluation — returns prediction and truth file paths
    Ypred_file, Ytruth_file = ntwk.evaluate(
        test_x, test_y,
        save_output=True,
        save_dir="evaluation_results/",
        prefix=flags.eval_model.replace('/', '_')
    )

    # Plot MSE loss distribution
    plotMSELossDistrib(Ypred_file, Ytruth_file, save_dir="evaluation_results/")
    print("Evaluation finished")


def evaluate_all(models_dir="models"):
    """
    Evaluate all models in the specified directory.
    """
    for file in os.listdir(models_dir):
        if os.path.isfile(os.path.join(models_dir, file, 'flags.obj')):
            evaluate_from_model(os.path.join(models_dir, file))
    return None


if __name__ == '__main__':
    model_name = "MLP/20250728_010501" 
    evaluate_from_model(model_name)
