import math
import tensorflow as tf

class Network:

    def __init__(self, dimensions, weights, biases):
        self.dimensions = dimensions
        self.weights = weights
        self.biases = biases

    def get_output_layer(self, inputs):
        # on first layer use provided inputs
        cur_layer_inputs = inputs

        for layer_ind, layer_dim in enumerate(self.dimensions[1:]):
            cur_layer_output = []
            for neuron in range(layer_dim):
                cur_layer_output.append(self._get_neuron_output(cur_layer_inputs, self.weights[layer_ind][neuron],
                                                                self.biases[layer_ind][neuron]))
            cur_layer_inputs = cur_layer_output
        return cur_layer_inputs

    def _get_neuron_output(self, inputs, weights, bias):
        return self.sigmoid(bias + sum([elems[0] * elems[1] for elems in zip(inputs, weights)]))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def squared_loss(expected, value):
        return (expected - value)**2

    def calculate_loss(self, inputs_with_results):
        sum_loss = 0.0

        for input_with_result in inputs_with_results:
            result = input_with_result[-1]
            inputs = input_with_result[:-1]
            sum_loss += self.squared_loss(result, self.get_output_layer(inputs)[0])
        return -sum_loss



def main():

    # example
    weights = [
        [[1, 2, 3], [1, 2, 3]],
        [[1, 2]]
    ]
    biases = [
        [1, 2],
        [1]
    ]

    # 3 input neurons, 2 hidden layer neurons, 1 output neuron
    netw = Network([3, 2, 1], weights, biases)
    #print(netw.get_output_layer([0, 1, 3]))

    and_weights = [
        [[10, 10]]
    ]
    and_biases = [
        [-15]
    ]

    and_netw = Network([2, 1], [[[10, 10]]], [[-15]])
    or_netw = Network([2, 1], [[[10, 10]]], [[-5]])
    not_netw = Network([1, 1], [-10], [[5]])
    not_and_netw = Network([2, 1], [[-10, -10]], [[15]])

    and_table = [[0, 0, 0],
                 [0, 1, 0],
                 [1, 0, 0],
                 [1, 1, 1]]

    or_table = [[0, 0, 0],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 1]]

    xor_table = [[0, 0, 0],
                 [0, 1, 1],
                 [1, 0, 1],
                 [1, 1, 0]]

    not_table = [[0, 1],
                 [1, 0]]

    print(and_netw.calculate_loss(and_table))


if __name__ == '__main__':
    main()
