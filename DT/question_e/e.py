import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import decisoin tree from scikit learn
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import sys
import os

# format : python {a/b/c/d/e/f}.py <train_data_path> <val_data_path> <test_data_path> <output_folder_path>
train_data_path = sys.argv[1]
val_data_path = sys.argv[2]
test_data_path = sys.argv[3]
output_folder_path = sys.argv[4]

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



max_depths = [15, 25, 35, 45]

train_acc = []
test_acc = []
best_test_acc = 0
best_val_acc = 0
best_val_depth = 0

for depth in max_depths:
    clf = DecisionTreeClassifier(max_depth=depth, criterion='entropy')
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    train_accuracy = 100*np.mean(y_train_pred == y_train)
    test_accuracy = 100*np.mean(y_test_pred == y_test)
    val_accuracy = 100*np.mean(clf.predict(X_val) == y_val)

    train_acc.append(train_accuracy)
    test_acc.append(test_accuracy)

    if test_accuracy > best_test_acc:
        best_test_acc = test_accuracy

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        best_val_depth = depth

    print(f"Depth: {depth}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")

print(f"Best Validation Accuracy: {best_val_acc:.2f}% at Depth: {best_val_depth}")
# Plotting train and test accuracy vs max_depth
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_acc, marker='o', label='Train Accuracy')
plt.plot(max_depths, test_acc, marker='o', label='Test Accuracy')
plt.title('Decision Tree: Train and Test Accuracy vs Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy (%)')
plt.xticks(max_depths)
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder_path, 'accuracy_vs_max_depth.png'))
plt.show()
plt.close()


ccp_alphas = [0.0, 0.0001, 0.0003, 0.0005]
train_acc_pruned = []
test_acc_pruned = []
best_val_acc_pruned = 0
best_val_alpha = 0.0

for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(criterion='entropy', ccp_alpha=alpha)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    train_accuracy = 100*np.mean(y_train_pred == y_train)
    test_accuracy = 100*np.mean(y_test_pred == y_test)
    val_accuracy = 100*np.mean(clf.predict(X_val) == y_val)

    train_acc_pruned.append(train_accuracy)
    test_acc_pruned.append(test_accuracy)

    if val_accuracy > best_val_acc_pruned:
        best_val_acc_pruned = val_accuracy
        best_val_alpha = alpha

    print(f"Alpha: {alpha}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")

print(f"Best Validation Accuracy after Pruning: {best_val_acc_pruned:.2f}% at Alpha: {best_val_alpha}")
# Plotting train and test accuracy vs ccp_alpha
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_acc_pruned, marker='o', label='Train Accuracy')
plt.plot(ccp_alphas, test_acc_pruned, marker='o', label='Test Accuracy')
plt.title('Decision Tree: Train and Test Accuracy vs CCP Alpha')
plt.xlabel('CCP Alpha')
plt.ylabel('Accuracy (%)')
plt.xticks(ccp_alphas)
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder_path, 'accuracy_vs_ccp_alpha.png'))
plt.show()
plt.close()