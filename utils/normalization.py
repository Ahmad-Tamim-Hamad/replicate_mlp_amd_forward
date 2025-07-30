# normalization.py

import numpy as np

# -------------------------------
# ✅ INPUT NORMALIZATION (features like geometry)
# -------------------------------

def normalize(x):
    """Symmetric input normalization: [-1, 1] range."""
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x_range = (x_max - x_min) / 2.0
    x_avg = (x_max + x_min) / 2.0
    x_norm = (x - x_avg) / (x_range + 1e-12)

    for i in range(x.shape[1]):
        print(f"In normalize_np, row {i}  your max is: {x_max[i]}")
        print(f"In normalize_np, row {i}  your min is: {x_min[i]}")
    
    return x_norm, x_max, x_min

def apply_normalization(x, x_max, x_min):
    """Apply same normalization parameters to new data."""
    x_range = (x_max - x_min) / 2.0
    x_avg = (x_max + x_min) / 2.0
    x_norm = (x - x_avg) / (x_range + 1e-12)
    return x_norm

def unnormalize(y_norm, y_max, y_min):
    """Undo symmetric normalization to get real values."""
    y_range = (y_max - y_min) / 2.0
    y_avg = (y_max + y_min) / 2.0
    y = y_norm * y_range + y_avg
    return y


def normalize_spectrum_columnwise(y):
    """Normalize spectrum per wavelength (column-wise), so global shape is preserved."""
    y_min = np.min(y, axis=0)
    y_max = np.max(y, axis=0)
    y_norm = (y - y_min) / (y_max - y_min + 1e-12)
    return y_norm, y_max, y_min

def unnormalize_spectrum_columnwise(y_norm, y_max, y_min):
    """Unnormalize the predicted spectrum."""
    return y_norm * (y_max - y_min + 1e-12) + y_min
