import numpy as np
import pandas as pd


data = pd.read_csv("data.csv")
X = data[['feature1', 'feature2']]
y1 = data['target1']
y2 = data['target2']

# Train Models
params = {'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.01, 'num_iterations': 500}
model1 = lgb.train(params, lgb.Dataset(X, label=y1))
model2 = lgb.train(params, lgb.Dataset(X, label=y2))

# Predictions
y1_pred = model1.predict(X)
y2_pred = model2.predict(X)
product_pred = y1_pred * y2_pred

# Evaluation
rmse1 = np.sqrt(mean_squared_error(y1, y1_pred))
rmse2 = np.sqrt(mean_squared_error(y2, y2_pred))

print(f"RMSE for target1: {rmse1:.3f}")
print(f"RMSE for target2: {rmse2:.3f}")

# Save Results
results = pd.DataFrame({'feature1': X['feature1'], 'target1_pred': y1_pred, 'target2_pred': y2_pred, 'product_pred': product_pred})
results.to_csv('results.csv', index=False)

