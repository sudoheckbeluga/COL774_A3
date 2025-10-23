import numpy as np
import pandas as pd
from collections import Counter

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None, children=None):
        self.feature = feature          # Index of the feature to split on
        self.threshold = threshold      # Threshold value for the split
        self.children = children if children is not None else []
        self.value = value              # Class label for leaf nodes, can also be used for non-leaf nodes in categorical splits
        self.is_leaf = False
        self.left = left
        self.right = right

class DecisionTree:
    def __init__(self, max_depth=None, criteria="entropy"):
        self.max_depth = max_depth
        self.root = None
        self.criteria = criteria

    def fit(self, X, y):
        parent_majority = None
        if len(y) > 0:
            parent_majority = self._majority_class(y)
        self.root = self._build_tree(X, y, parent_majority_class=parent_majority)

    def _build_tree(self, X, y, depth=0, parent_majority_class=None):
        if len(y) == 0:
            leaf_node = DecisionTreeNode(value=parent_majority_class)
            leaf_node.is_leaf = True
            return leaf_node

        current_majority = self._majority_class(y)

        if len(set(y)) == 1 or depth == self.max_depth:  # Pure node
            leaf_value = current_majority
            leaf_node = DecisionTreeNode(value=leaf_value)
            leaf_node.is_leaf = True
            return leaf_node

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            leaf_node = DecisionTreeNode(value=current_majority)
            leaf_node.is_leaf = True
            return leaf_node

        # Split the data
        if best_threshold is None:
            # Categorical feature
            unique_values = np.unique(X[:, best_feature])
            children = []
            for value in unique_values:
                indices = X[:, best_feature] == value
                child_node = self._build_tree(X[indices], y[indices], depth + 1, parent_majority_class=current_majority)
                children.append((value, child_node))
            return DecisionTreeNode(feature=best_feature,value=current_majority, children=children)
        else:
            # Numerical feature
            left_indices = X[:, best_feature] <= best_threshold
            right_indices = X[:, best_feature] > best_threshold
            left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1, parent_majority_class=current_majority)
            right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1, parent_majority_class=current_majority)
            return DecisionTreeNode(feature=best_feature, threshold=best_threshold,value=current_majority, left=left_child, right=right_child)


    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gain = -1e-9
        for feature in range(X.shape[1]):
            if self._is_categorical(X[:, feature]):
                unique_values = np.unique(X[:, feature])
                if len(unique_values) == 1:
                    continue  # Cannot split on constant categorical feature
                if self.criteria == "entropy":
                    gain = self.entropy_gain(X, y, feature, unique_values, None)
                else:
                    gain = self.gini_gain(X, y, feature, unique_values, None)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = None
            else:
                values = X[:, feature].astype(float)
                # Only skip if all values are identical
                if np.all(values == values[0]):
                    continue
                unique_vals = np.unique(values)
                threshold = np.median(values)
                if self.criteria == "entropy":
                    gain = self.entropy_gain(X, y, feature, unique_vals, threshold)
                else:
                    gain = self.gini_gain(X, y, feature, unique_vals, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    
    def _is_categorical(self, feature_column):
        if len(feature_column) == 0:
            return False
        return isinstance(feature_column[0], str)
    
    def entropy_gain(self, X, y, feature, unique_values, threshold):
        parent_entropy = self.entropy(y)
        if threshold is None:
            child_entropy = 0
            for value in unique_values:
                indices = X[:, feature] == value
                child_entropy += (len(y[indices]) / len(y)) * self.entropy(y[indices])
            return parent_entropy - child_entropy
        else:
            left_indices = X[:, feature] <= threshold
            right_indices = X[:, feature] > threshold
            if not np.any(left_indices) or not np.any(right_indices):
                return 0

            left_entropy = self.entropy(y[left_indices])
            right_entropy = self.entropy(y[right_indices])
            child_entropy = (len(y[left_indices]) * left_entropy + len(y[right_indices]) * right_entropy) / len(y)
            return parent_entropy - child_entropy

    def entropy(self, y):
        # Calculate the entropy of a label array
        if len(y) == 0:
            return 0
        class_counts = Counter(y)
        probabilities = [count / len(y) for count in class_counts.values()]
        return -sum(p * np.log2(p) for p in probabilities if p > 0)
    
    def _majority_class(self, y):
        # Return the majority class label
        return Counter(y).most_common(1)[0][0]
    
    def gini_gain(self, X, y, feature, unique_values, threshold):
        parent_gini = self.gini_index(y)
        if threshold is None:
            child_gini = 0
            for value in unique_values:
                indices = X[:, feature] == value
                child_gini += (len(y[indices]) / len(y)) * self.gini_index(y[indices])
            return parent_gini - child_gini
        else:
            left_indices = X[:, feature] <= threshold
            right_indices = X[:, feature] > threshold
            if not np.any(left_indices) or not np.any(right_indices):
                return 0

            left_gini = self.gini_index(y[left_indices])
            right_gini = self.gini_index(y[right_indices])
            child_gini = (len(y[left_indices]) * left_gini + len(y[right_indices]) * right_gini) / len(y)
            return parent_gini - child_gini

    def gini_index(self, y):
        # Calculate the Gini index of a label array
        if len(y) == 0:
            return 0
        class_counts = Counter(y)
        probabilities = [count / len(y) for count in class_counts.values()]
        return 1 - sum(p ** 2 for p in probabilities)

    def predict(self, X):
        return np.array([self._sample_predict(sample, self.root) for sample in X])
    
    def _sample_predict(self,sample,node):
        if node.is_leaf:
            return node.value
        
        if node.threshold is None:  # Categorical split
            # Check if the sample's value exists in children
            child_values = [v for v, _ in node.children]
            if sample[node.feature] not in child_values:
                return node.value
            for value, child in node.children:
                if sample[node.feature] == value:
                    return self._sample_predict(sample, child)
        else:  # Numerical split
            if sample[node.feature] <= node.threshold:
                return self._sample_predict(sample, node.left)
            else:
                return self._sample_predict(sample, node.right)
    

    def calculate_accuracy(self, X, y):
        """Helper to calculate accuracy in percentage."""
        if len(y) == 0:
            return 0.0
        y_pred = self.predict(X)
        return 100 * np.mean(y_pred == y)

    def count_nodes(self):
        """Counts the total number of nodes in the tree."""
        return self._count_nodes_recursive(self.root)

    def _count_nodes_recursive(self, node):
        """Recursive helper for count_nodes."""
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        
        count = 1  # Count this internal node
        if node.threshold is None:  # Categorical
            for _, child in node.children:
                count += self._count_nodes_recursive(child)
        else:  # Numerical
            count += self._count_nodes_recursive(node.left)
            count += self._count_nodes_recursive(node.right)
        return count

    def _get_all_non_leaf_nodes(self):
        """Get all non-leaf nodes using Breadth-First Search."""
        nodes = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if node is None or node.is_leaf:
                continue
            
            nodes.append(node)  # Add this non-leaf node
            
            if node.threshold is None:  # Categorical
                for _, child in node.children:
                    queue.append(child)
            else:  # Numerical
                queue.append(node.left)
                queue.append(node.right)
        return nodes

    def prune(self, X_val, y_val, X_train, y_train, X_test, y_test):
        """
        Performs post-pruning on the tree using a validation set.
        Returns a pandas DataFrame with metrics at each pruning step.
        """
        metrics = []
        
        # Get initial metrics before any pruning
        current_val_acc = self.calculate_accuracy(X_val, y_val)
        metrics.append({
            'nodes': self.count_nodes(),
            'train_acc': self.calculate_accuracy(X_train, y_train),
            'val_acc': current_val_acc,
            'test_acc': self.calculate_accuracy(X_test, y_test)
        })

        while True:
            non_leaf_nodes = self._get_all_non_leaf_nodes()
            if not non_leaf_nodes:
                break  # No more nodes to prune

            best_new_acc = -1
            best_node_to_prune = None

            # Find the best node to prune
            for node in non_leaf_nodes:
                # 1. Store original state
                original_state = {
                    'feature': node.feature,
                    'threshold': node.threshold,
                    'left': node.left,
                    'right': node.right,
                    'children': node.children,
                    'is_leaf': node.is_leaf
                }
                
                # 2. Prune (convert to leaf)
                # node.value already stores the majority class from training
                node.is_leaf = True
                node.feature = None
                node.threshold = None
                node.left = None
                node.right = None
                node.children = []

                # 3. Calculate new validation accuracy
                new_val_acc = self.calculate_accuracy(X_val, y_val)

                # 4. Check if this is the best prune so far
                if new_val_acc > best_new_acc:
                    best_new_acc = new_val_acc
                    best_node_to_prune = node

                # 5. Restore original state for next iteration
                node.is_leaf = original_state['is_leaf']
                node.feature = original_state['feature']
                node.threshold = original_state['threshold']
                node.left = original_state['left']
                node.right = original_state['right']
                node.children = original_state['children']
            
            # After checking all nodes, get the baseline accuracy
            current_val_acc = self.calculate_accuracy(X_val, y_val)
            
            # 6. Check if the best prune is actually an improvement
            # We prune even if accuracy is equal, to get a simpler tree
            if best_new_acc >= current_val_acc and best_node_to_prune is not None:
                # Permanently prune the best node
                node = best_node_to_prune
                node.is_leaf = True
                node.feature = None
                node.threshold = None
                node.left = None
                node.right = None
                node.children = []
                
                # Record metrics for this new pruned tree
                metrics.append({
                    'nodes': self.count_nodes(),
                    'train_acc': self.calculate_accuracy(X_train, y_train),
                    'val_acc': best_new_acc,
                    'test_acc': self.calculate_accuracy(X_test, y_test)
                })
            else:
                # No pruning operation improved accuracy, so stop
                break
                
        return pd.DataFrame(metrics)