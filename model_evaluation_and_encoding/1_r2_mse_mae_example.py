import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# Create our student exam data
data = {
    'hours_studied': [2, 5, 3, 8, 1, 6],
    'actual_score': [65, 80, 70, 95, 60, 85],
    'predicted_score': [70, 85, 75, 100, 65, 90]
}

# Convert to DataFrame
df = pd.DataFrame(data)
print("Student Exam Data:")
print(df)

# Calculate errors
df['error'] = df['predicted_score'] - df['actual_score']
df['squared_error'] = df['error']**2
print("\nErrors:")
print(df[['error', 'squared_error']])

# Calculate MAE
mae = np.mean(np.abs(df['error']))
print(f"\nMAE = {mae}")

# Calculate MSE
mse = np.mean(df['squared_error'])
print(f"MSE = {mse}")

# Calculate R²
mean_actual = np.mean(df['actual_score'])
sst = sum((df['actual_score'] - mean_actual)**2)  # Total sum of squares
ssr = sum(df['squared_error'])  # Sum of squared residuals
r_squared = 1 - (ssr/sst)
print(f"R² = {r_squared:.4f} or {r_squared*100:.2f}%")

# Verify calculations with scikit-learn
sk_mae = mean_absolute_error(df['actual_score'], df['predicted_score'])
sk_mse = mean_squared_error(df['actual_score'], df['predicted_score'])
sk_r2 = r2_score(df['actual_score'], df['predicted_score'])

print("\nVerification with scikit-learn:")
print(f"MAE = {sk_mae}")
print(f"MSE = {sk_mse}")
print(f"R² = {sk_r2:.4f}")

# Create a scatterplot to visualize data and predictions
plt.figure(figsize=(10, 6))
plt.scatter(df['hours_studied'], df['actual_score'], color='blue', label='Actual Scores')
plt.scatter(df['hours_studied'], df['predicted_score'], color='red', label='Predicted Scores')

# Connect actual and predicted with lines to show errors
for i in range(len(df)):
    plt.plot([df['hours_studied'][i], df['hours_studied'][i]], 
             [df['actual_score'][i], df['predicted_score'][i]], 
             'k--', alpha=0.5)

plt.title('Exam Scores vs. Hours Studied')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.grid(True, alpha=0.3)

# Create annotation for metrics
metrics_text = f"MAE = {mae}\nMSE = {mse}\nR² = {r_squared:.4f}"
plt.annotate(metrics_text, xy=(0.05, 0.05), xycoords='axes fraction', 
             bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

plt.show()
