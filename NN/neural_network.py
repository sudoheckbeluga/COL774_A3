import numpy as np

class NeuralNetwork:
    def __init__(self, mini_batch_size, n_features, hidden_layers, n_classes, activation='sigmoid', learning_rate=0.01):
        self.mini_batch_size = mini_batch_size
        self.n_features = n_features
        self.hidden_layers = hidden_layers
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self.activation_function = activation
        self._initialize_parameters()

    def _initialize_parameters(self):
        layer_sizes = [self.n_features] + self.hidden_layers + [self.n_classes]
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            bias_vector = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def _sigmoid_derivative(self, a):
        return a * (1 - a)
    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _relu(self, z):
        return np.maximum(0, z)
    def _relu_derivative(self, a):
        return (a > 0).astype(float)
    
    def forward_prop(self, X):
        a = X
        activations = [a]
        zs = []
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(a, w) + b
            zs.append(z)
            if self.activation_function == 'sigmoid':
                a = self._sigmoid(z)
            else:
                a = self._relu(z)
            activations.append(a)
        # final layer with softmax
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        zs.append(z)
        a = self._softmax(z)
        activations.append(a)
        return activations, zs

    def compute_loss(self, Y_pred, Y_true):
        # cross entropy loss = - sum(y_true * log(y_pred)) / m
        m = Y_true.shape[0]
        log_likelihood = -np.log(Y_pred[range(m), Y_true] + 1e-9)
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backward_prop(self, X, Y_true, activations, zs):
        m = X.shape[0]
        weights_grads = [np.zeros_like(w) for w in self.weights]
        biases_grads = [np.zeros_like(b) for b in self.biases]

        # output layer gradient
        delta = activations[-1].copy()
        delta[range(m), Y_true] -= 1
        delta /= m
        weights_grads[-1] = np.dot(activations[-2].T, delta)
        biases_grads[-1] = np.sum(delta, axis=0, keepdims=True)

        # hidden layers gradients
        for l in range(len(self.hidden_layers), 0, -1):
            if self.activation_function == 'sigmoid':
                delta = np.dot(delta, self.weights[l].T) * self._sigmoid_derivative(activations[l])
            else:
                delta = np.dot(delta, self.weights[l].T) * self._relu_derivative(activations[l])
            weights_grads[l - 1] = np.dot(activations[l - 1].T, delta)
            biases_grads[l - 1] = np.sum(delta, axis=0, keepdims=True)

        return weights_grads, biases_grads

    def update_parameters(self, grads):
        # update weights and biases using gradients
        weights_grads, biases_grads = grads
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weights_grads[i]
            self.biases[i] -= self.learning_rate * biases_grads[i]

    def train(self, X_train, Y_train, epochs=100):
        m = X_train.shape[0]

        for epoch in range(epochs):
            # 🔀 Shuffle data at the start of each epoch
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            Y_train = Y_train[indices]

            # 🔄 Iterate through mini-batches
            for i in range(0, m, self.mini_batch_size):
                X_batch = X_train[i:i+self.mini_batch_size]
                Y_batch = Y_train[i:i+self.mini_batch_size]

                activations, zs = self.forward_prop(X_batch)
                grads = self.backward_prop(X_batch, Y_batch, activations, zs)
                self.update_parameters(grads)

            # 📉 Compute loss after each epoch (optional: full dataset or last batch)
            activations, _ = self.forward_prop(X_train)
            loss = self.compute_loss(activations[-1], Y_train)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
  

    def predict(self, X):
        activations, _ = self.forward_prop(X)
        return np.argmax(activations[-1], axis=1)
    

        
