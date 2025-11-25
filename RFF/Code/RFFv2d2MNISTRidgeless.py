import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore")

NOISE_LEVEL = 0.20   
TOTAL_SAMPLES = 12500 
TEST_SIZE = 0.2
INPUT_DIM = 28 * 28
N_CLASSES = 10
SEED=94

N_TRAIN_SAMPLES = int(TOTAL_SAMPLES * (1 - TEST_SIZE))
N_TEST_SAMPLES = int(TOTAL_SAMPLES * TEST_SIZE)

center = N_TRAIN_SAMPLES
COMPONENT_RANGE = np.unique(np.concatenate([
    np.linspace(100, center - 200, 5, dtype=int),
    np.arange(center - 200, center + 201, 50),  
    np.linspace(center + 201, 6 * center, 15, dtype=int)
]))

print("Loading and preparing MNIST dataset...")
(X_full, y_full), (X_test_orig, y_test_orig) = tf.keras.datasets.mnist.load_data()

X_all = np.concatenate([X_full, X_test_orig]) / 255.0
y_all = np.concatenate([y_full, y_test_orig])
X_all_flat = X_all.reshape(-1, INPUT_DIM)
y_all_flat = y_all.flatten()

X_sample, _, y_sample, _ = train_test_split(
    X_all_flat, y_all_flat, 
    train_size=TOTAL_SAMPLES, 
    random_state=SEED, 
    stratify=y_all_flat
)

X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, 
    test_size=N_TEST_SAMPLES, 
    random_state=SEED, 
    stratify=y_sample
)

print(f"Introducing {NOISE_LEVEL * 100:.0f}% label noise to training set...")
n_noise = int(NOISE_LEVEL * N_TRAIN_SAMPLES)
noise_indices = np.random.choice(N_TRAIN_SAMPLES, size=n_noise, replace=False)
original_labels = y_train[noise_indices]
random_offsets = np.random.randint(1, N_CLASSES, size=n_noise)
y_train[noise_indices] = (original_labels + random_offsets) % N_CLASSES

y_train_ohe = to_categorical(y_train, num_classes=N_CLASSES)
y_test_ohe = to_categorical(y_test, num_classes=N_CLASSES)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gamma = 0.02 

def compute_weights_ridgeless(X, Y):
      N, D = X.shape

    if N >= D:

        W, _, _, _ = np.linalg.lstsq(X, Y)
        return W

    else:
        K = X @ X.T 

        c, _, _, _ = np.linalg.lstsq(K, Y)

        W = X.T @ c
        return W

loss_train, loss_test = [], []
mse_train, mse_test = [], []
weight_norms = []

start_time = time.time()

for n_components in COMPONENT_RANGE:

    rff_sampler = RBFSampler(gamma=gamma, n_components=n_components, random_state=SEED)
    X_train_rff = rff_sampler.fit_transform(X_train_scaled).astype(np.float32)
    X_test_rff = rff_sampler.transform(X_test_scaled).astype(np.float32)

    W_hat = compute_weights_ridgeless(X_train_rff, y_train_ohe)

    w_norm = np.linalg.norm(W_hat)
    weight_norms.append(w_norm)

    y_train_pred_scores = X_train_rff @ W_hat
    y_test_pred_scores = X_test_rff @ W_hat

    y_train_pred_labels = np.argmax(y_train_pred_scores, axis=1)
    y_test_pred_labels = np.argmax(y_test_pred_scores, axis=1)

    train_loss = 1.0 - accuracy_score(y_train, y_train_pred_labels)
    test_loss = 1.0 - accuracy_score(y_test, y_test_pred_labels)
    loss_train.append(train_loss)
    loss_test.append(test_loss)

    train_mse = mean_squared_error(y_train_ohe, y_train_pred_scores)
    test_mse = mean_squared_error(y_test_ohe, y_test_pred_scores)
    mse_train.append(train_mse)
    mse_test.append(test_mse)

    print(f"Feats: {n_components:5d} | Test Err: {test_loss:.4f} | MSE: {test_mse:.2f} | ||W||: {w_norm:.2f}")

print(f"\n Experiment finished in {time.time() - start_time:.2f} seconds.")

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(18, 5)) 

plt.subplot(1, 3, 1)
plt.plot(COMPONENT_RANGE, loss_train, label='Train 0-1', linestyle='--', alpha=0.6)
plt.plot(COMPONENT_RANGE, loss_test, label='Test 0-1', color='tab:orange', linewidth=2)
plt.axvline(x=N_TRAIN_SAMPLES, color='k', linestyle=':', alpha=0.5, label='Threshold')
plt.xlabel('Number of Features')
plt.ylabel('Classification Error')
plt.title(f'Double Descent (Noise {NOISE_LEVEL*100}%)')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(COMPONENT_RANGE, mse_train, label='Train MSE', linestyle='--', alpha=0.6)
plt.plot(COMPONENT_RANGE, mse_test, label='Test MSE', color='tab:green', linewidth=2)
plt.axvline(x=N_TRAIN_SAMPLES, color='k', linestyle=':', alpha=0.5)
plt.yscale('log') 
plt.xlabel('Number of Features')
plt.ylabel('MSE (Log Scale)')
plt.title('Mean Squared Error')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(COMPONENT_RANGE, weight_norms, color='tab:red', linewidth=2, label='||W||')
plt.axvline(x=N_TRAIN_SAMPLES, color='k', linestyle=':', alpha=0.5, label='Threshold')
plt.yscale('log') 
plt.xlabel('Number of Features')
plt.ylabel('Norm (Log Scale)')
plt.title('Norm of Weights')
plt.legend()

plt.tight_layout()
plt.show()

data_to_save = {
    'Components': COMPONENT_RANGE,
    'Train 0-1 Loss': loss_train,
    'Test 0-1 Loss': loss_test,
    'Train MSE': mse_train,
    'Test MSE': mse_test,
    'Weight Norm': weight_norms
}
df = pd.DataFrame(data_to_save)

csv_filename = "nsamp=10000_nlvl=20per_gamma=d02_seed="+str(SEED)+".csv"
df.to_csv(csv_filename, index=False)
print(f" Data points saved to {csv_filename}")
