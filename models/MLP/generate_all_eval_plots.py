import numpy as np
import matplotlib.pyplot as plt
import os

save_dir = 'evaluation_results'
prefix = 'MLP_20250728_010501'  # ✅ correct one

Ypred_file = os.path.join(save_dir, f'test_Ypred_{prefix}.csv')
Ytruth_file = os.path.join(save_dir, f'test_Ytruth_{prefix}.csv')

Ypred = np.loadtxt(Ypred_file, delimiter=None)
Ytruth = np.loadtxt(Ytruth_file, delimiter=None)

indices = [0, 10, 100, 500]
for idx in indices:
    plt.figure()
    plt.plot(Ytruth[idx], label='Truth')
    plt.plot(Ypred[idx], label='Prediction')
    plt.title(f'Spectrum Comparison (Sample #{idx})')
    plt.xlabel('Frequency Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'spectrum_comparison_{idx}.png'))

abs_error = np.abs(Ytruth - Ypred)
mean_abs_error = np.mean(abs_error, axis=1)
plt.figure(figsize=(10, 5))
plt.hist(mean_abs_error, bins=100)
plt.title(f'Histogram of Mean Absolute Errors (MAE)\nAvg MAE = {np.mean(mean_abs_error):.4f}')
plt.xlabel('Mean Absolute Error')
plt.ylabel('Count')
plt.grid(True)
plt.savefig(os.path.join(save_dir, f'mae_distribution_{prefix}.png'))

mse_error = np.mean((Ytruth - Ypred) ** 2, axis=1)
plt.figure(figsize=(10, 5))
plt.hist(mse_error, bins=100)
plt.title(f'Histogram of Mean Squared Errors (MSE)\nAvg MSE = {np.mean(mse_error):.4f}')
plt.xlabel('Mean Squared Error')
plt.ylabel('Count')
plt.grid(True)
plt.savefig(os.path.join(save_dir, f'mse_distribution_{prefix}.png'))

plt.figure(figsize=(8, 6))
plt.bar([prefix], [np.mean(mse_error)], yerr=[np.std(mse_error)], capsize=10)
plt.ylabel('Mean Squared Error')
plt.title(f'MSE Summary for {prefix}')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'mse_summary_errorbar_{prefix}.png'))

print(f'Final evaluation MSE for {prefix}: {np.mean(mse_error):.4f} ± {np.std(mse_error):.4f}')

print("All evaluation plots saved to:", save_dir)
