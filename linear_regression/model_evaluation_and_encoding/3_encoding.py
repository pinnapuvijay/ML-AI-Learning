import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Create a sample dataset with educational degrees and salaries
np.random.seed(42)  # for reproducibility

data = {
    'degree': ['High School', 'Bachelor\'s', 'Master\'s', 'PhD', 
               'High School', 'Bachelor\'s', 'Master\'s', 'PhD',
               'High School', 'Bachelor\'s', 'Master\'s', 'PhD'],
    'experience_years': [5, 3, 7, 10, 2, 5, 3, 15, 8, 12, 6, 4],
    'salary': [45000, 60000, 75000, 90000, 
               40000, 65000, 70000, 95000,
               50000, 70000, 80000, 85000]
}

df = pd.DataFrame(data)
print("Original Data:")
print(df.head())

# 1. Label Encoding
le = LabelEncoder()
df['degree_label'] = le.fit_transform(df['degree'])
print("\nAfter Label Encoding:")
print(df[['degree', 'degree_label']].drop_duplicates())

# 2. One-Hot Encoding
# Using scikit-learn's ColumnTransformer with OneHotEncoder
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(sparse_output=False), ['degree'])],
    remainder='passthrough'
)
encoded_array = ct.fit_transform(df[['degree', 'experience_years']])

# Get the feature names
encoded_feature_names = ct.get_feature_names_out(['degree', 'experience_years'])

# Convert to DataFrame
df_encoded = pd.DataFrame(encoded_array, columns=encoded_feature_names)
print("\nAfter One-Hot Encoding (first few columns):")
print(df_encoded.iloc[:, :5].head())

# Add salary to both DataFrames for modeling
df_encoded['salary'] = df['salary']
df_label = df[['degree_label', 'experience_years', 'salary']]

# 3. Compare performance with linear regression
# Split features and target for both encoding methods
X_label = df_label[['degree_label', 'experience_years']]
X_onehot = df_encoded.drop('salary', axis=1)
y = df['salary']

# Fit linear regression models
lr_label = LinearRegression().fit(X_label, y)
lr_onehot = LinearRegression().fit(X_onehot, y)

# Predictions
pred_label = lr_label.predict(X_label)
pred_onehot = lr_onehot.predict(X_onehot)

# Calculate MSE
mse_label = mean_squared_error(y, pred_label)
mse_onehot = mean_squared_error(y, pred_onehot)

print("\nLinear Regression Performance:")
print(f"MSE with Label Encoding: {mse_label:.2f}")
print(f"MSE with One-Hot Encoding: {mse_onehot:.2f}")

# 4. Compare performance with decision tree
# Fit decision tree models
dt_label = DecisionTreeRegressor(random_state=42).fit(X_label, y)
dt_onehot = DecisionTreeRegressor(random_state=42).fit(X_onehot, y)

# Predictions
dt_pred_label = dt_label.predict(X_label)
dt_pred_onehot = dt_onehot.predict(X_onehot)

# Calculate MSE
dt_mse_label = mean_squared_error(y, dt_pred_label)
dt_mse_onehot = mean_squared_error(y, dt_pred_onehot)

print("\nDecision Tree Performance:")
print(f"MSE with Label Encoding: {dt_mse_label:.2f}")
print(f"MSE with One-Hot Encoding: {dt_mse_onehot:.2f}")

# 5. Visualize results
plt.figure(figsize=(12, 6))

# Linear Regression Chart
plt.subplot(1, 2, 1)
models = ['Label Encoding', 'One-Hot Encoding']
lr_mse = [mse_label, mse_onehot]
plt.bar(models, lr_mse, color=['blue', 'orange'])
plt.title('Linear Regression MSE')
plt.ylabel('Mean Squared Error')
plt.grid(axis='y', alpha=0.3)

# Decision Tree Chart
plt.subplot(1, 2, 2)
dt_mse = [dt_mse_label, dt_mse_onehot]
plt.bar(models, dt_mse, color=['blue', 'orange'])
plt.title('Decision Tree MSE')
plt.ylabel('Mean Squared Error')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# 6. Visual comparison of encodings
plt.figure(figsize=(12, 6))

# Label Encoding Visualization
plt.subplot(1, 2, 1)
plt.scatter(df['degree_label'], df['salary'], alpha=0.7)
plt.title('Label Encoding: Degree vs. Salary')
plt.xlabel('Degree (Encoded as Integer)')
plt.ylabel('Salary')
plt.xticks([0, 1, 2, 3], ['High School', 'Bachelor\'s', 'Master\'s', 'PhD'])
plt.grid(True, alpha=0.3)

# One-Hot Encoding Matrix Visualization (heatmap-style)
plt.subplot(1, 2, 2)
degree_columns = [col for col in df_encoded.columns if 'degree' in col]
one_hot_matrix = df_encoded[degree_columns].iloc[:4].values  # Just show the first 4 rows
plt.imshow(one_hot_matrix, cmap='Blues', aspect='auto')
plt.title('One-Hot Encoding Matrix')
plt.xlabel('Degree Category')
plt.ylabel('First 4 Data Points')
plt.xticks(range(4), ['HS', 'Bachelor\'s', 'Master\'s', 'PhD'])
plt.yticks(range(4), ['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4'])
for i in range(4):
    for j in range(4):
        plt.text(j, i, int(one_hot_matrix[i, j]), ha='center', va='center')

plt.tight_layout()
plt.show()
