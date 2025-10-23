import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decision_tree import DecisionTree 

# --- 1. Load Data ---
# format : python run_pruning.py <train_path> <val_path> <test_path> <output_folder>
if len(sys.argv) != 5:
    print("Usage: python run_pruning.py <train_path> <val_path> <test_path> <output_folder>")
    sys.exit(1)

train_data_path = sys.argv[1]
val_data_path = sys.argv[2]
test_data_path = sys.argv[3]
output_folder_path = sys.argv[4]

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Load dataframes
train_df = pd.read_csv(train_data_path)
val_df = pd.read_csv(val_data_path)
test_df = pd.read_csv(test_data_path)

# Separate features (X) and labels (y)
y_train = train_df.pop('result').values
X_train = train_df
y_val = val_df.pop('result').values
X_val = val_df
y_test = test_df.pop('result').values
X_test = test_df

print("Data loaded.")

# --- 2. One-Hot Encoding (Consistent across all sets) ---
# Identify categorical columns from the *training data*
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Apply one-hot encoding
X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)
X_val_encoded = pd.get_dummies(X_val, columns=categorical_cols, drop_first=False)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=False)

# Align columns: Ensure all datasets have the exact same columns as the training set
X_train_cols = X_train_encoded.columns
X_val_encoded = X_val_encoded.reindex(columns=X_train_cols, fill_value=0)
X_test_encoded = X_test_encoded.reindex(columns=X_train_cols, fill_value=0)

# Convert to numpy arrays
X_train_np = X_train_encoded.values
X_val_np = X_val_encoded.values
X_test_np = X_test_encoded.values

print("One-hot encoding complete.")

# --- 3. Train and Prune ---
max_depths = [15, 25, 35, 45] # Using depths from your part (b)
all_metrics = {}

for depth in max_depths:
    print(f"--- Training tree with max_depth={depth} ---")
    tree = DecisionTree(max_depth=depth)
    tree.fit(X_train_np, y_train)
    
    print(f"Initial (unpruned) nodes: {tree.count_nodes()}")
    print(f"Initial val accuracy: {tree.calculate_accuracy(X_val_np, y_val):.4f}%")

    # Prune the tree and get metrics
    # We pass all datasets to get all 3 accuracies at each step
    metrics_df = tree.prune(X_val_np, y_val, X_train_np, y_train, X_test_np, y_test)
    all_metrics[depth] = metrics_df
    
    print(f"Pruning complete for depth {depth}.")
    print(f"Final nodes: {metrics_df.iloc[-1]['nodes']}")
    print(f"Final (best) val accuracy: {metrics_df['val_acc'].max():.4f}%")
    print("-" * 30)
    if (depth==35):
        y_test_df = pd.read_csv(test_data_path)
        y_test_values = y_test_df.pop('result').values
        y_test_pred = tree.predict(X_test_np)
        output_df = pd.DataFrame({'result': y_test_pred})
        output_df.to_csv(os.path.join(output_folder_path, 'depth_35_predictions.csv'), index=False)
        print("Test Predictions saved for depth 35 tree")

# ==========================================================
# == MODIFIED PLOTTING SECTION START HERE ==
# ==========================================================

# --- 4. Plot Results (Separate for each tree) ---
print("\nGenerating separate plots for each initial max_depth...")

for depth, metrics_df in all_metrics.items():
    # Create a new figure for each depth
    plt.figure(figsize=(12, 8))
    
    # Sort by nodes (descending) to show pruning progression
    metrics_df_sorted = metrics_df.sort_values(by='nodes', ascending=False)
    
    plt.plot(metrics_df_sorted['nodes'], metrics_df_sorted['train_acc'], marker='.', linestyle='--', label='Train Accuracy')
    plt.plot(metrics_df_sorted['nodes'], metrics_df_sorted['val_acc'], marker='o', markersize=5, linestyle='-', label='Validation Accuracy')
    plt.plot(metrics_df_sorted['nodes'], metrics_df_sorted['test_acc'], marker='x', linestyle=':', label='Test Accuracy')

    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Pruning Analysis (Started at max_depth={depth})', fontsize=14)
    plt.legend()
    plt.grid(True)
    # Invert X-axis so "unpruned" (more nodes) is on the left
    plt.gca().invert_xaxis() 
    plt.tight_layout()

    # Save the plot with a unique name
    plot_path = os.path.join(output_folder_path, f'pruning_plot_depth_{depth}.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Show the plot
    plt.show()

print("All plots generated.")