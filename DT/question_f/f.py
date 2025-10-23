import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import os

# format : python {a/b/c/d/e/f}.py <train_data_path> <val_data_path> <test_data_path> <output_folder_path>
train_data_path = sys.argv[1]
val_data_path = sys.argv[2]
test_data_path = sys.argv[3]
output_folder_path = sys.argv[4]

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

X_train = pd.read_csv(train_data_path)
y_train = X_train.pop('result').values

X_val = pd.read_csv(val_data_path)
y_val = X_val.pop('result').values

X_test = pd.read_csv(test_data_path)
y_test = X_test.pop('result').values

# One-hot encoding for categorical features
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")

X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)
X_val_encoded = pd.get_dummies(X_val, columns=categorical_cols, drop_first=False)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=False)

# Align columns to ensure all datasets have the same features
X_train_cols = X_train_encoded.columns
X_val_encoded = X_val_encoded.reindex(columns=X_train_cols, fill_value=0)
X_test_encoded = X_test_encoded.reindex(columns=X_train_cols, fill_value=0)

# Convert to numpy arrays
X_train = X_train_encoded.values
X_val = X_val_encoded.values
X_test = X_test_encoded.values

print(f"Number of features after encoding: {X_train.shape[1]}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Test samples: {X_test.shape[0]}\n")

# Define parameter grid for grid search
param_grid = {
    'n_estimators': [50, 150, 250, 350],
    'max_features': [0.1, 0.3, 0.5, 0.7, 0.9],
    'min_samples_split': [2, 4, 6, 8, 10],
    'criterion': ['entropy'],
    'oob_score': [True],  # Enable out-of-bag scoring
}

print("Parameter Grid:")
for key, values in param_grid.items():
    print(f"  {key}: {values}")
print()

# Combine train and validation for grid search
X_train_val = np.vstack([X_train, X_val])
y_train_val = np.concatenate([y_train, y_val])

# Create validation indices for GridSearchCV
# We want to use validation set for selecting best parameters
train_indices = np.arange(len(X_train))
val_indices = np.arange(len(X_train), len(X_train_val))
cv_split = [(train_indices, val_indices)]

print("Starting Grid Search...")
print(f"Total combinations to try: {len(param_grid['n_estimators']) * len(param_grid['max_features']) * len(param_grid['min_samples_split'])}")
print()

# Perform grid search
grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=cv_split,  # Use our custom train/val split
    scoring='accuracy',
    verbose=2,
    n_jobs=-1  # Use all available cores
)

grid_search.fit(X_train_val, y_train_val)

print("\n" + "="*60)
print("GRID SEARCH COMPLETED")
print("="*60)

# Get best parameters
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("\nBest Parameters:")
for key, value in best_params.items():
    print(f"  {key}: {value}")
print(f"\nBest Validation Accuracy from Grid Search: {best_score * 100:.2f}%")

# Train the best model on training data only
print("\n" + "="*60)
print("TRAINING BEST MODEL ON TRAINING DATA")
print("="*60)

best_rf = RandomForestClassifier(**best_params)
best_rf.fit(X_train, y_train)

# Calculate accuracies
y_train_pred = best_rf.predict(X_train)
y_val_pred = best_rf.predict(X_val)
y_test_pred = best_rf.predict(X_test)

train_accuracy = 100 * np.mean(y_train_pred == y_train)
val_accuracy = 100 * np.mean(y_val_pred == y_val)
test_accuracy = 100 * np.mean(y_test_pred == y_test)
oob_accuracy = 100 * best_rf.oob_score_

print(f"\nFinal Results with Best Parameters:")
print(f"  Training Accuracy:        {train_accuracy:.2f}%")
print(f"  Out-of-Bag (OOB) Accuracy: {oob_accuracy:.2f}%")
print(f"  Validation Accuracy:      {val_accuracy:.2f}%")
print(f"  Test Accuracy:            {test_accuracy:.2f}%")

# Save results to file
results_file = os.path.join(output_folder_path, 'random_forest_results.txt')
with open(results_file, 'w') as f:
    f.write("Random Forest - Grid Search Results\n")
    f.write("="*60 + "\n\n")
    f.write("Best Parameters:\n")
    for key, value in best_params.items():
        f.write(f"  {key}: {value}\n")
    f.write(f"\nAccuracies:\n")
    f.write(f"  Training Accuracy:        {train_accuracy:.2f}%\n")
    f.write(f"  Out-of-Bag (OOB) Accuracy: {oob_accuracy:.2f}%\n")
    f.write(f"  Validation Accuracy:      {val_accuracy:.2f}%\n")
    f.write(f"  Test Accuracy:            {test_accuracy:.2f}%\n")

print(f"\nResults saved to: {results_file}")

# Save predictions
predictions_df = pd.DataFrame({'result': y_test_pred})
predictions_file = os.path.join(output_folder_path, 'test_predictions.csv')
predictions_df.to_csv(predictions_file, index=False)
print(f"Test predictions saved to: {predictions_file}")

# Convert grid search results to DataFrame for analysis
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv(os.path.join(output_folder_path, 'grid_search_results.csv'), index=False)

# Plot: n_estimators vs accuracy (fixing other parameters at best values)
print("\nGenerating plots...")

# Filter results for best max_features and min_samples_split
best_max_features = best_params['max_features']
best_min_samples_split = best_params['min_samples_split']

filtered_results = cv_results[
    (cv_results['param_max_features'] == best_max_features) &
    (cv_results['param_min_samples_split'] == best_min_samples_split)
]

plt.figure(figsize=(10, 6))
plt.plot(filtered_results['param_n_estimators'], 
         filtered_results['mean_test_score'] * 100, 
         marker='o', linewidth=2, markersize=8)
plt.xlabel('Number of Estimators', fontsize=12)
plt.ylabel('Validation Accuracy (%)', fontsize=12)
plt.title(f'Random Forest: Validation Accuracy vs n_estimators\n(max_features={best_max_features}, min_samples_split={best_min_samples_split})', 
          fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_folder_path, 'n_estimators_vs_accuracy.png'), dpi=150, bbox_inches='tight')
plt.close()

# Plot: max_features vs accuracy (fixing other parameters at best values)
best_n_estimators = best_params['n_estimators']

filtered_results2 = cv_results[
    (cv_results['param_n_estimators'] == best_n_estimators) &
    (cv_results['param_min_samples_split'] == best_min_samples_split)
]

plt.figure(figsize=(10, 6))
plt.plot(filtered_results2['param_max_features'], 
         filtered_results2['mean_test_score'] * 100, 
         marker='s', linewidth=2, markersize=8, color='green')
plt.xlabel('Max Features', fontsize=12)
plt.ylabel('Validation Accuracy (%)', fontsize=12)
plt.title(f'Random Forest: Validation Accuracy vs max_features\n(n_estimators={best_n_estimators}, min_samples_split={best_min_samples_split})', 
          fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_folder_path, 'max_features_vs_accuracy.png'), dpi=150, bbox_inches='tight')
plt.close()

# Plot: min_samples_split vs accuracy (fixing other parameters at best values)
filtered_results3 = cv_results[
    (cv_results['param_n_estimators'] == best_n_estimators) &
    (cv_results['param_max_features'] == best_max_features)
]

plt.figure(figsize=(10, 6))
plt.plot(filtered_results3['param_min_samples_split'], 
         filtered_results3['mean_test_score'] * 100, 
         marker='^', linewidth=2, markersize=8, color='red')
plt.xlabel('Min Samples Split', fontsize=12)
plt.ylabel('Validation Accuracy (%)', fontsize=12)
plt.title(f'Random Forest: Validation Accuracy vs min_samples_split\n(n_estimators={best_n_estimators}, max_features={best_max_features})', 
          fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_folder_path, 'min_samples_split_vs_accuracy.png'), dpi=150, bbox_inches='tight')
plt.close()

# Create a summary comparison plot
plt.figure(figsize=(10, 6))
accuracies = [train_accuracy, oob_accuracy, val_accuracy, test_accuracy]
labels = ['Training', 'Out-of-Bag', 'Validation', 'Test']
colors = ['blue', 'orange', 'green', 'red']

plt.bar(labels, accuracies, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Random Forest: Accuracy Comparison (Best Model)', fontsize=14)
plt.ylim([min(accuracies) - 5, 100])
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (label, acc) in enumerate(zip(labels, accuracies)):
    plt.text(i, acc + 0.5, f'{acc:.2f}%', ha='center', fontsize=11, fontweight='bold')

plt.savefig(os.path.join(output_folder_path, 'accuracy_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

print("All plots saved successfully!")
print("\n" + "="*60)
print("RANDOM FOREST ANALYSIS COMPLETE")
print("="*60)

