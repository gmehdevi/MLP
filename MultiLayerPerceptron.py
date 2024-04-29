import numpy as np
import pickle




class Optimizer:
    def __init__(self, parameters):
        self.parameters = parameters

    def update_params(self, grads):
        pass

class Stochastic(Optimizer):
    def __init__(self, parameters, learning_rate=0.01):
        super().__init__(parameters)
        self.learning_rate = learning_rate

    def update_params(self, grads):
        for param, grad in zip(self.parameters, grads):
            param[0] -= self.learning_rate * grad[0]  # Update weights
            param[1] -= self.learning_rate * grad[1]  # Update biases

class Momentum(Optimizer):
    def __init__(self, parameters, learning_rate=0.01, momentum=0.9):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = [(np.zeros_like(w), np.zeros_like(b)) for w, b in parameters]

    def update_params(self, grads):
        for (param, v), grad in zip(zip(self.parameters, self.v), grads):
            v[0] = self.momentum * v[0] - self.learning_rate * grad[0]
            v[1] = self.momentum * v[1] - self.learning_rate * grad[1]
            param[0] += v[0]
            param[1] += v[1]

class Adagrad(Optimizer):
    def __init__(self, parameters, learning_rate=0.01, epsilon=1e-8):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.G = [(np.zeros_like(w), np.zeros_like(b)) for w, b in parameters]

    def update_params(self, grads):
        for (param, G), grad in zip(zip(self.parameters, self.G), grads):
            G[0] += grad[0] ** 2
            G[1] += grad[1] ** 2
            param[0] -= self.learning_rate * grad[0] / (np.sqrt(G[0]) + self.epsilon)
            param[1] -= self.learning_rate * grad[1] / (np.sqrt(G[1]) + self.epsilon)

class Layer:
    def relu(self, Z):
        return np.maximum(0, Z)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def __init__(self, input_size, output_size, activation='relu', init_method='he'):

        activation_functions = {
            'relu': self.relu,
            'sigmoid': self.sigmoid,
            'softmax': self.softmax
        }

        param_inits = {
            'he': lambda input_size, output_size: (np.random.randn(output_size, input_size) * np.sqrt(2. / input_size), np.zeros((output_size, 1))),
            'xavier': lambda input_size, output_size: (np.random.randn(output_size, input_size) * np.sqrt(1. / input_size), np.zeros((output_size, 1))),
            'random': lambda input_size, output_size: (np.random.randn(output_size, input_size), np.zeros((output_size, 1)))
        }

        self.W, self.b = param_inits[init_method](input_size, output_size)
        self.activation = activation_functions[activation]

    def forward(self, A_prev):
        Z = np.dot(self.W, A_prev) + self.b
        return Z

    def backward(self, dZ, A_prev):
        m = A_prev.shape[1]
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.W.T, dZ)
        return dW, db, dA_prev


class MultiLayerPerceptron:
    def __init__(self, layer_sizes, activations, init_methods, optimizer='stochastic', learning_rate=0.01, momentum=0.9, seed=42, loss_function='log'):
        loss_functions = {
            'cross_entropy': self.cross_entropy_loss,
            'hinge': self.hinge_loss,
            'log': self.log_loss
        }
        np.random.seed(seed)
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1], activations[i], init_methods[i]) for i in range(len(layer_sizes)-1)]
        parameters = [(layer.W, layer.b) for layer in self.layers]
        self.optimizer = self.create_optimizer(optimizer, parameters, learning_rate, momentum)
        self.loss = loss_functions[loss_function]

    def create_optimizer(self, optimizer_type, parameters, learning_rate, momentum):
        if optimizer_type == 'stochastic':
            return Stochastic(parameters, learning_rate)
        elif optimizer_type == 'Momentum':
            return Momentum(parameters, learning_rate, momentum)
        elif optimizer_type == 'adagrad':
            return Adagrad(parameters, learning_rate)
        else:
            raise ValueError("Unsupported optimizer")

    def update_parameters(self, grads):
        self.optimizer.update_params(grads)

    def cross_entropy_loss(self, AL, Y):
        m = Y.shape[1]
        cost = -np.sum(np.log(AL) * Y) / m
        return cost

    def hinge_loss(self, AL, Y):
        m = Y.shape[1]
        cost = np.sum(np.maximum(0, 1 - Y * AL)) / m
        return cost

    def log_loss(self, AL, Y):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m
        return cost

    def compute_cost(self, AL, Y):
        return self.loss(AL, Y)

    def forward_propagation(self, X):
        A = X
        caches = []
        for layer in self.layers[:-1]:
            A_prev = A
            Z = layer.forward(A_prev)
            A = layer.activation(Z)
            caches.append((A_prev, Z))

        A_prev = A
        Z = self.layers[-1].forward(A_prev)
        AL = self.layers[-1].activation(Z)
        caches.append((A_prev, Z))
        return AL, caches

    def backward_propagation(self, AL, Y, caches):
        grads = {}
        L = len(self.layers)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = AL - Y
        A_prev, Z = caches[-1]
        dZ = dAL
        grads["dW" + str(L)], grads["db" + str(L)], dA_prev = self.layers[-1].backward(dZ, A_prev)

        for l in reversed(range(L - 1)):
            A_prev, Z = caches[l]
            dZ = dA_prev * (Z > 0)
            grads["dW" + str(l + 1)], grads["db" + str(l + 1)], dA_prev = self.layers[l].backward(dZ, A_prev)

        return grads

    def update_parameters(self, grads, learning_rate):
        L = len(self.layers)
        for l in range(L):
            self.layers[l].W -= learning_rate * grads['dW' + str(l + 1)]
            self.layers[l].b -= learning_rate * grads['db' + str(l + 1)]

    def compute_accuracy(self, predictions, labels):
        return np.mean(np.argmax(predictions, axis=0) == np.argmax(labels, axis=0))

    def fit(self, X, Y, learning_rate=0.01, epochs=100, X_val=None, Y_val=None, patience=5):
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        best_val_loss = float('inf')
        no_improve_epoch = 0

        for epoch in range(epochs):
            AL, caches = self.forward_propagation(X)
            grads = self.backward_propagation(AL, Y, caches)
            self.update_parameters(grads, learning_rate)

            loss = self.compute_cost(AL, Y)
            accuracy = self.compute_accuracy(AL, Y)
            history['loss'].append(loss)
            history['accuracy'].append(accuracy)

            if X_val is not None and Y_val is not None:
                AL_val, _ = self.forward_propagation(X_val)
                val_loss = self.compute_cost(AL_val, Y_val)
                val_accuracy = self.compute_accuracy(AL_val, Y_val)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_epoch = 0
                else:
                    no_improve_epoch += 1

                if patience >= 0 and no_improve_epoch >= patience:
                    break

            print(f"Epoch {epoch + 1}/{epochs} - loss: {loss:.4f} - accuracy: {accuracy:.4f} - val_loss: {val_loss:.4f} - val_accuracyuracy: {val_accuracy:.4f}")

        return history

    def predict(self, X):
        AL, _ = self.forward_propagation(X)
        return AL

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        loss = self.compute_cost(predictions, Y)
        accuracy = self.compute_accuracy(predictions, Y)
        return loss, accuracy

    def save_model(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def load_model(self, path):
        with open(path, 'rb') as file:
            model = pickle.load(file)
        self.__dict__.update(model.__dict__)
        
def create_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    mlp = MultiLayerPerceptron([],[],[])
    mlp.__dict__.update(model.__dict__)
    return mlp