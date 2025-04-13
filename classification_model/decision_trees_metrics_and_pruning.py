# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
np.random.seed(42)

# Create the dataset
def create_exam_dataset():
    """Create a dataset of students with study hours and exam results."""
    data = {
        'student_id': list(range(1, 51)),
        'hours_studied': [
            2.5, 7.0, 4.5, 3.0, 5.5, 2.0, 8.0, 3.5, 6.5, 1.5,  # 1-10
            4.0, 6.0, 2.0, 7.5, 3.5, 5.0, 1.0, 4.5, 2.5, 7.0,  # 11-20
            3.0, 6.0, 2.5, 5.5, 1.5, 4.0, 8.5, 3.0, 5.0, 7.5,  # 21-30
            3.2, 6.8, 2.8, 4.8, 3.7, 5.3, 1.8, 7.3, 2.7, 4.7,  # 31-40
            3.9, 6.2, 2.3, 5.8, 3.4, 4.9, 2.1, 7.8, 3.6, 5.2   # 41-50
        ],
        'exam_score': [
            68, 89, 77, 69, 85, 62, 94, 73, 87, 58,  # 1-10
            75, 84, 65, 93, 72, 81, 52, 76, 67, 88,  # 11-20
            70, 86, 66, 82, 59, 74, 95, 71, 80, 91,  # 21-30
            71, 88, 69, 79, 72, 83, 61, 90, 68, 78,  # 31-40
            74, 86, 64, 84, 73, 79, 63, 92, 74, 82   # 41-50
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add pass/fail column (pass if score >= 75)
    df['pass_fail'] = df['exam_score'].apply(lambda x: 'Pass' if x >= 75 else 'Fail')
    
    return df

# Create the dataset
exam_df = create_exam_dataset()

# Display the first few rows of the dataset
print("Exam Dataset:")
print(exam_df.head(10))
print("\nDataset Information:")
print(exam_df.describe())

# Check pass/fail distribution
print("\nPass/Fail Distribution:")
print(exam_df['pass_fail'].value_counts())

# Encode the target variable (pass/fail)
le = LabelEncoder()
exam_df['pass_fail_encoded'] = le.fit_transform(exam_df['pass_fail'])
# 'Pass' = 1, 'Fail' = 0

# Define features and target
X = exam_df[['hours_studied']]
y = exam_df['pass_fail_encoded']

# Split the data into training, validation, and test sets
# First 30 for training, next 10 for validation, last 10 for testing
X_train = X.iloc[:30]
y_train = y.iloc[:30]
X_val = X.iloc[30:40]
y_val = y.iloc[30:40]
X_test = X.iloc[40:50]
y_test = y.iloc[40:50]

print("\nData Split:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Function to calculate Gini impurity
def gini_impurity(y):
    """Calculate Gini impurity for a set of labels."""
    if len(y) == 0:
        return 0
    p = np.sum(y) / len(y)  # Proportion of positive class (Pass)
    return p * (1 - p) * 2  # Gini impurity = 2 * p * (1-p) for binary

# Calculate Gini impurity for training set
gini_root = gini_impurity(y_train)
print(f"\nGini impurity for entire training set: {gini_root:.3f}")

# Find optimal split manually
def find_best_split(X, y):
    """Find the best split point based on Gini impurity."""
    best_gain = -1
    best_split = None
    
    # Try all possible values as split points
    unique_values = sorted(X['hours_studied'].unique())
    
    for i in range(len(unique_values) - 1):
        split_point = (unique_values[i] + unique_values[i+1]) / 2
        
        # Split data
        left_mask = X['hours_studied'] <= split_point
        right_mask = ~left_mask
        
        # Calculate weighted Gini impurity
        left_gini = gini_impurity(y[left_mask.values])
        right_gini = gini_impurity(y[right_mask.values])
        
        n_left = sum(left_mask)
        n_right = sum(right_mask)
        n_total = len(y)
        
        weighted_gini = (n_left/n_total * left_gini) + (n_right/n_total * right_gini)
        gain = gini_root - weighted_gini
        
        print(f"Split at hours_studied <= {split_point:.1f}: Gain = {gain:.3f}")
        
        if gain > best_gain:
            best_gain = gain
            best_split = split_point
    
    return best_split, best_gain

# Find best split point manually
best_split, best_gain = find_best_split(X_train, y_train)
print(f"\nBest split: hours_studied <= {best_split:.1f} with Gini gain = {best_gain:.3f}")

# Train pruned decision tree (depth=1)
pruned_tree = DecisionTreeClassifier(max_depth=1, random_state=42)
pruned_tree.fit(X_train, y_train)

# Train complex decision tree (deeper tree)
complex_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=42)
complex_tree.fit(X_train, y_train)

# Visualize the pruned tree
plt.figure(figsize=(12, 6))
plot_tree(pruned_tree, filled=True, feature_names=['hours_studied'], 
          class_names=['Fail', 'Pass'], proportion=True, fontsize=10)
plt.title('Pruned Decision Tree (max_depth=1)')
plt.tight_layout()
plt.savefig('pruned_tree.png')
plt.show()

# Visualize the complex tree
plt.figure(figsize=(15, 10))
plot_tree(complex_tree, filled=True, feature_names=['hours_studied'], 
          class_names=['Fail', 'Pass'], proportion=True, fontsize=8)
plt.title('Complex Decision Tree (max_depth=5)')
plt.tight_layout()
plt.savefig('complex_tree.png')
plt.show()

# Function to evaluate model and calculate metrics
def evaluate_model(model, X, y, set_name):
    """Evaluate model performance and print metrics."""
    y_pred = model.predict(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # Generate confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    print(f"\n{set_name} Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fail', 'Pass'], 
                yticklabels=['Fail', 'Pass'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {set_name}')
    plt.tight_layout()
    plt.savefig(f'{model.__class__.__name__}_{set_name}_cm.png')
    plt.show()
    
    return accuracy, precision, recall, f1

# Evaluate pruned tree on validation set
print("\n--- Pruned Tree Evaluation ---")
pruned_val_metrics = evaluate_model(pruned_tree, X_val, y_val, "Validation Set")

# Evaluate complex tree on validation set
print("\n--- Complex Tree Evaluation ---")
complex_val_metrics = evaluate_model(complex_tree, X_val, y_val, "Validation Set")

# Evaluate pruned tree on test set
pruned_test_metrics = evaluate_model(pruned_tree, X_test, y_test, "Test Set")

# Evaluate complex tree on test set
complex_test_metrics = evaluate_model(complex_tree, X_test, y_test, "Test Set")

# Compare models using a bar chart
def plot_comparison(pruned_metrics, complex_metrics, set_name):
    """Create a bar chart comparing model performance."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    pruned_values = list(pruned_metrics)
    complex_values = list(complex_metrics)
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, pruned_values, width, label='Pruned Tree')
    rects2 = ax.bar(x + width/2, complex_values, width, label='Complex Tree')
    
    ax.set_ylabel('Score')
    ax.set_title(f'Model Comparison on {set_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f'model_comparison_{set_name}.png')
    plt.show()

# Plot comparison of models on validation set
plot_comparison(pruned_val_metrics, complex_val_metrics, "Validation Set")

# Plot comparison of models on test set
plot_comparison(pruned_test_metrics, complex_test_metrics, "Test Set")

# Visualize the decision boundaries
def plot_decision_boundary(model, title):
    """Plot the decision boundary for a model."""
    # Create a meshgrid of points to plot the decision boundary
    x_min, x_max = 0.5, 9
    hours_studied = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    
    # Get predictions for each point in the meshgrid
    predictions = model.predict(hours_studied)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    for i, label in enumerate(['Fail', 'Pass']):
        mask = y_train == i
        plt.scatter(X_train.iloc[mask.values]['hours_studied'], 
                    [i * 0.05 + 0.5 for _ in range(sum(mask))],
                    label=f'Training: {label}', alpha=0.7)
    
    # Plot decision boundary
    plt.plot(hours_studied, predictions, 'r-', linewidth=2, label='Decision Boundary')
    
    # Add a vertical line at the split point
    plt.axvline(x=best_split, color='g', linestyle='--', 
                label=f'Split Point: hours_studied <= {best_split:.1f}')
    
    plt.xlabel('Hours Studied')
    plt.yticks([0, 1], ['Fail', 'Pass'])
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.show()

# Plot decision boundaries
plot_decision_boundary(pruned_tree, "Pruned Tree Decision Boundary")

# Create a scatterplot of all data points
plt.figure(figsize=(12, 8))
plt.scatter(exam_df[exam_df['pass_fail'] == 'Fail']['hours_studied'], 
            exam_df[exam_df['pass_fail'] == 'Fail']['exam_score'],
            color='red', label='Fail', alpha=0.7)
plt.scatter(exam_df[exam_df['pass_fail'] == 'Pass']['hours_studied'], 
            exam_df[exam_df['pass_fail'] == 'Pass']['exam_score'],
            color='green', label='Pass', alpha=0.7)

# Add vertical line at the best split point
plt.axvline(x=best_split, color='blue', linestyle='--', 
            label=f'Split Point: hours_studied = {best_split:.1f}')

# Add horizontal line at passing score
plt.axhline(y=75, color='black', linestyle=':', label='Passing Score = 75')

plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Exam Scores vs. Hours Studied')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('exam_scores_vs_hours.png')
plt.show()

# Print a summary of the findings
print("\n--- Summary of Findings ---")
print(f"Best split point: hours_studied = {best_split:.1f}")
print("\nValidation Set Performance:")
print(f"Pruned Tree: Accuracy = {pruned_val_metrics[0]:.2f}, Precision = {pruned_val_metrics[1]:.2f}, Recall = {pruned_val_metrics[2]:.2f}, F1 = {pruned_val_metrics[3]:.2f}")
print(f"Complex Tree: Accuracy = {complex_val_metrics[0]:.2f}, Precision = {complex_val_metrics[1]:.2f}, Recall = {complex_val_metrics[2]:.2f}, F1 = {complex_val_metrics[3]:.2f}")

print("\nTest Set Performance:")
print(f"Pruned Tree: Accuracy = {pruned_test_metrics[0]:.2f}, Precision = {pruned_test_metrics[1]:.2f}, Recall = {pruned_test_metrics[2]:.2f}, F1 = {pruned_test_metrics[3]:.2f}")
print(f"Complex Tree: Accuracy = {complex_test_metrics[0]:.2f}, Precision = {complex_test_metrics[1]:.2f}, Recall = {complex_test_metrics[2]:.2f}, F1 = {complex_test_metrics[3]:.2f}")

if pruned_test_metrics[0] > complex_test_metrics[0]:
    print("\nThe pruned tree outperforms the complex tree on the test set, demonstrating the benefits of simplicity and avoiding overfitting.")
else:
    print("\nThe complex tree performs better on the test set, suggesting that the additional complexity captures important patterns in the data.")
