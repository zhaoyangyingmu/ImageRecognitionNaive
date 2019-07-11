import numpy as np
import matplotlib.pyplot as plt
from util.network_elements import Network
from util.network_util import NetworkUtil
from util.image_data import ImageData


class CompareBias:
    def __init__(self):
        # set up network
        net_config = [784, 14, -1]
        learning_rate = 0.1
        weight = -0.10
        l_bias = list((np.arange(11) - 5) * 0.1)
        networks = []
        for bias in l_bias:
            networks.append(Network(net_config, learning_rate, weight, bias))
        print("networks ready!")
        self.l_bias = l_bias
        self.networks = networks
        self.weight = weight
        self.l_ll = []
        pass

    def training(self):
        l_bias = self.l_bias
        networks = self.networks
        get_hit_rate = NetworkUtil.get_hit_rate
        weight = self.weight
        # training
        [x_train, y_train] = ImageData.get_train_data()
        print("training data ready!")

        l_ll = [0.0] * len(l_bias)
        for i in range(len(l_bias)):
            l_ll[i] = []

        for e in range(100):
            for i in range(287):
                idx = i * 10
                for j in range(len(networks)):
                    loss = networks[j].train(x_train[idx:idx + 10, :], y_train[idx:idx + 10, :])
                    l_ll[j].append(loss)
            print("epoch", e)
            for i in range(len(l_bias)):
                hit_rate = get_hit_rate(networks[i])
                print("hit rate =", hit_rate, " with weight =", weight, "bias =", l_bias[i])
        self.l_ll = l_ll

    def show_result(self):
        l_bias = self.l_bias
        l_ll = self.l_ll
        plt.title('loss with different biases')
        for i in range(len(l_bias)):
            x = np.arange(1, len(l_ll[i]) + 1)
            label_str = 'bias = ' + str(l_bias[i])
            plt.plot(x[100:], l_ll[i][100:], label=(label_str))
        plt.legend()
        plt.xlabel('iteration times')
        plt.ylabel('loss')
        plt.show()

    def get_best_network(self):
        best_network = NetworkUtil.get_best_network(self.networks)
        return best_network
