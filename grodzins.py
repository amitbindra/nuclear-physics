import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===========================
# 1️⃣ Generate Sample Data
# ===========================
np.random.seed(42)
data_size = 500  # Number of samples

data = pd.DataFrame({
    'Z': np.random.randint(10, 100, data_size),  # Atomic Number
    'N': np.random.randint(10, 150, data_size),  # Neutron Number
    'A': np.random.randint(20, 250, data_size),  # Mass Number
    'S_p': np.random.uniform(0, 10, data_size),  # Separation Energy (Protons)
    'S_n': np.random.uniform(0, 10, data_size),  # Separation Energy (Neutrons)
    'beta': np.random.uniform(0, 1, data_size),  # Deformation Parameter
    'B': np.random.uniform(0, 100, data_size),  # Nuclear Binding Energy
    'P': np.random.uniform(0, 1, data_size),  # Parity
    'nu_p': np.random.randint(0, 10, data_size),  # Proton Number in a Shell
    'nu_n': np.random.randint(0, 10, data_size),  # Neutron Number in a Shell
    'I2': np.random.uniform(0, 5, data_size),  # Nuclear Spin

    # Target variables (simulated based on physics-inspired relations)
    'E(2^+_1)': np.random.uniform(0.1, 2.0, data_size) + 0.01 * np.random.randn(data_size),
    'B(E2; 0^+ -> 2^+)': np.random.uniform(10, 500, data_size) + 0.5 * np.random.randn(data_size)
})

# ===========================
# 2️⃣ Define Features & Targets
# ===========================
input_features = ['Z', 'N', 'A', 'S_p', 'S_n', 'beta', 'B', 'P', 'nu_p', 'nu_n', 'I2']
target_E2 = 'E(2^+_1)'
target_BE2 = 'B(E2; 0^+ -> 2^+)'

X = data[input_features]
y_E2 = data[target_E2]
y_BE2 = data[target_BE2]

# ===========================
# 3️⃣ Train-Test Split
# ===========================
X_train, X_val, y_E2_train, y_E2_val, y_BE2_train, y_BE2_val = train_test_split(
    X, y_E2, y_BE2, test_size=0.1, random_state=42
)

# ===========================
# 4️⃣ Train LightGBM Models
# ===========================

# Model for E(2^+_1)
train_data_E2 = lgb.Dataset(X_train, label=y_E2_train)
params_E2 = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 10,
    'max_depth': -1,
    'num_iterations': 500,
    'learning_rate': 0.01,
    'verbose': -1
}
model_E2 = lgb.train(params_E2, train_data_E2)

# Model for B(E2; 0^+ -> 2^+)
train_data_BE2 = lgb.Dataset(X_train, label=y_BE2_train)
params_BE2 = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 10,
    'max_depth': -1,
    'num_iterations': 500,
    'learning_rate': 0.01,
    'verbose': -1
}
model_BE2 = lgb.train(params_BE2, train_data_BE2)

# ===========================
# 5️⃣ Predictions
# ===========================
y_E2_pred = model_E2.predict(X_val)
y_BE2_pred = model_BE2.predict(X_val)

# Grodzins Product
grodzins_product = y_E2_pred * y_BE2_pred

# ===========================
# 6️⃣ Evaluation Metrics
# ===========================
rmse_E2 = np.sqrt(mean_squared_error(y_E2_val, y_E2_pred))
rmse_BE2 = np.sqrt(mean_squared_error(y_BE2_val, y_BE2_pred))
r2_E2 = r2_score(y_E2_val, y_E2_pred)
r2_BE2 = r2_score(y_BE2_val, y_BE2_pred)
mae_E2 = mean_absolute_error(y_E2_val, y_E2_pred)
mae_BE2 = mean_absolute_error(y_BE2_val, y_BE2_pred)

print(f"RMSE for E(2^+_1): {rmse_E2:.3f}")
print(f"RMSE for B(E2; 0^+ -> 2^+): {rmse_BE2:.3f}")
print(f"R² Score for E(2^+_1): {r2_E2:.3f}")
print(f"R² Score for B(E2; 0^+ -> 2^+): {r2_BE2:.3f}")
print(f"MAE for E(2^+_1): {mae_E2:.3f}")
print(f"MAE for B(E2; 0^+ -> 2^+): {mae_BE2:.3f}")

# ===========================
# 7️⃣ Visualization
# ===========================

# Scatter Plot (Actual vs. Predicted)
plt.figure(figsize=(6, 6))
plt.scatter(y_E2_val, y_E2_pred, alpha=0.7, edgecolors='k')
plt.plot([min(y_E2_val), max(y_E2_val)], [min(y_E2_val), max(y_E2_val)], 'r--')
plt.xlabel("Actual E(2^+_1)")
plt.ylabel("Predicted E(2^+_1)")
plt.title("Actual vs. Predicted E(2^+_1)")
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_BE2_val, y_BE2_pred, alpha=0.7, edgecolors='k')
plt.plot([min(y_BE2_val), max(y_BE2_val)], [min(y_BE2_val), max(y_BE2_val)], 'r--')
plt.xlabel("Actual B(E2; 0^+ -> 2^+)")
plt.ylabel("Predicted B(E2; 0^+ -> 2^+)")
plt.title("Actual vs. Predicted B(E2; 0^+ -> 2^+)")
plt.show()

# ===========================
# 8️⃣ Save Results
# ===========================
# ===========================
# 8️⃣ Save Results (Including Z, N, A)
# ===========================
# ===========================
# 8️⃣ Save Results (Including Z, N, A)
# ===========================
results = pd.DataFrame({
    'Z': X_val.reset_index(drop=True)['Z'],
    'N': X_val.reset_index(drop=True)['N'],
    'A': X_val.reset_index(drop=True)['A'],
    'E(2^+_1)_true': y_E2_val.reset_index(drop=True),
    'E(2^+_1)_pred': y_E2_pred,
    'B(E2; 0^+ -> 2^+)_true': y_BE2_val.reset_index(drop=True),
    'B(E2; 0^+ -> 2^+)_pred': y_BE2_pred,
    'Grodzins_product': grodzins_product
})

# Save to CSV
results.to_csv('grodzins_product_results.csv', index=False)

print("✅ Grodzins product calculation completed. Results saved to 'grodzins_product_results.csv'.")
