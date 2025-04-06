import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Original data with one predictor (hours studied)
data = {
    'hours_studied': [2, 5, 3, 8, 1, 6],
    'actual_score': [65, 80, 70, 95, 60, 85]
}

df = pd.DataFrame(data)
n = len(df)  # number of observations

# Fit a linear regression model to get predictions
model1 = LinearRegression()
X1 = df[['hours_studied']]
y = df['actual_score']
model1.fit(X1, y)
predictions1 = model1.predict(X1)

# Calculate R² for model with one predictor
r2_1 = r2_score(y, predictions1)
print(f"Model with one predictor (hours studied):")
print(f"R² = {r2_1:.4f}")

# Calculate adjusted R² for model with one predictor
p1 = 1  # number of predictors
adj_r2_1 = 1 - ((1 - r2_1) * (n - 1) / (n - p1 - 1))
print(f"Adjusted R² = {adj_r2_1:.4f}")

# Now let's add a second predictor (hours slept)
# We'll simulate this data for demonstration
np.random.seed(42)  # for reproducibility
df['hours_slept'] = np.random.normal(7, 1, size=n)  # random sleep hours

# Fit a new model with both predictors
model2 = LinearRegression()
X2 = df[['hours_studied', 'hours_slept']]
model2.fit(X2, y)
predictions2 = model2.predict(X2)

# Calculate R² for model with two predictors
r2_2 = r2_score(y, predictions2)
print(f"\nModel with two predictors (hours studied + hours slept):")
print(f"R² = {r2_2:.4f}")

# Calculate adjusted R² for model with two predictors
p2 = 2  # number of predictors
adj_r2_2 = 1 - ((1 - r2_2) * (n - 1) / (n - p2 - 1))
print(f"Adjusted R² = {adj_r2_2:.4f}")

# Compare the models
print("\nComparison:")
print(f"Model 1: R² = {r2_1:.4f}, Adjusted R² = {adj_r2_1:.4f}")
print(f"Model 2: R² = {r2_2:.4f}, Adjusted R² = {adj_r2_2:.4f}")

if adj_r2_1 > adj_r2_2:
    print("\nThe simpler model (hours studied only) is better according to adjusted R²")
else:
    print("\nThe more complex model (hours studied + hours slept) is better according to adjusted R²")

# Visualize R² vs Adjusted R² for both models
plt.figure(figsize=(10, 6))
models = ['Model 1 (One Predictor)', 'Model 2 (Two Predictors)']
r2_values = [r2_1, r2_2]
adj_r2_values = [adj_r2_1, adj_r2_2]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, r2_values, width, label='R²')
plt.bar(x + width/2, adj_r2_values, width, label='Adjusted R²')

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('R² vs Adjusted R² Comparison')
plt.xticks(x, models)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)

for i, v in enumerate(r2_values):
    plt.text(i - width/2, v + 0.02, f'{v:.4f}', ha='center')
    
for i, v in enumerate(adj_r2_values):
    plt.text(i + width/2, v + 0.02, f'{v:.4f}', ha='center')

plt.show()
