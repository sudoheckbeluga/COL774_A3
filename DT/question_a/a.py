import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decision_tree import DecisionTree

# format : python {a/b/c/d/e/f}.py <train_data_path> <val_data_path> <test_data_path> <output_folder_path>
train_data_path = sys.argv[1]
val_data_path = sys.argv[2]
test_data_path = sys.argv[3]
output_folder_path = sys.argv[4]


X_train = pd.read_csv(train_data_path)
y_train = X_train.pop('result').values
X_train = X_train.values

X_val = pd.read_csv(val_data_path)
y_val = X_val.pop('result').values
X_val = X_val.values

X_test = pd.read_csv(test_data_path)
y_test = X_test.pop('result').values
X_test = X_test.values

max_depths = [5, 10, 15, 20]
train_acc = []
test_acc = []
best_tree = None
best_test_acc = 0

for depth in max_depths:
    tree = DecisionTree(max_depth=depth)
    tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)

    train_accuracy = 100*np.mean(y_train_pred == y_train)
    test_accuracy = 100*np.mean(y_test_pred == y_test)
    if test_accuracy > best_test_acc:
        best_test_acc = test_accuracy
        best_tree = tree
    train_acc.append(train_accuracy)
    test_acc.append(test_accuracy)

    print(f"Depth {depth}: Train Acc = {train_accuracy:.4f} %, Test Acc = {test_accuracy:.4f} %")

# Plot accuracies
plt.figure(figsize=(8,5))
plt.plot(max_depths, train_acc, marker='o', label='Train Accuracy')
plt.plot(max_depths, test_acc, marker='o', label='Test Accuracy')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy vs Max Depth')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_folder_path, 'accuracy_vs_max_depth.png'))
plt.show()
# for math test accuracy, save the predicitions for Validation set in csv file. the file should have a single column 'result' containing the predicted labels
y_test = pd.read_csv(val_data_path)
y_test = y_test.pop('result').values
y_test_pred = best_tree.predict(X_val)
output_df = pd.DataFrame({'result': y_test_pred})
output_df.to_csv(os.path.join(output_folder_path, 'test_predictions.csv'), index=False)
print("Test Predictions saved for best tree")
print(f"Validation Accuracy: {np.mean(y_val == y_test_pred):.4f} %")

