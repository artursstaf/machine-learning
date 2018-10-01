import math


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
        return [round(item, 1) for item in cur_layer_inputs]

    def _get_neuron_output(self, inputs, weights, bias):
        return self.sigmoid(bias + sum([elems[0] * elems[1] for elems in zip(inputs, weights)]))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))


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
    print(netw.get_output_layer([0, 1, 3]))

    and_weights = [
        [[10, 10]]
    ]
    and_biases = [
        [-15]
    ]
    and_netw = Network([2, 1], and_weights, and_biases)
    print(and_netw.get_output_layer([1, 1]))
    print(and_netw.get_output_layer([0, 1]))
    print(and_netw.get_output_layer([1, 0]))
    print(and_netw.get_output_layer([0, 0]))


if __name__ == '__main__':
    main()
