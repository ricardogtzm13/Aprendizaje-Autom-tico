import numpy as np

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size + 1)  # +1 bias

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels, epochs):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += (label - prediction) * inputs
                self.weights[0] += (label - prediction)

# Ejemplo de uso
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])

perceptron = Perceptron(input_size = 2)
perceptron.train(training_inputs, labels, epochs=10)

# Prueba de la compuerta AND
print("Resultado de la compuerta AND:")
print("0 AND 0 =", perceptron.predict([0, 0]))
print("0 AND 1 =", perceptron.predict([0, 1]))
print("1 AND 0 =", perceptron.predict([1, 0]))
print("1 AND 1 =", perceptron.predict([1, 1]))
