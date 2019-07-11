import numpy as np
import matplotlib.pyplot as plt
from util.network_elements import Network
from util.image_data import ImageData
from util.network_util import NetworkUtil


class CompareWeight:
    def __init__(self):
        # set up network
        net_config = [784,14,-1]
        learning_rate = 0.1
        l_weights = list((np.arange(11)-5)*0.1)
        bias = 0.5
        networks = []
        for weight in l_weights:
            networks.append(Network(net_config,learning_rate,weight,bias))
        print("networks ready!")
        self.l_weights = l_weights
        self.networks = networks
        self.bias = bias
        self.l_ll = []
        pass

    def training(self):
        l_weights = self.l_weights
        networks = self.networks
        get_hit_rate = NetworkUtil.get_hit_rate
        bias = self.bias
        # training
        [x_train, y_train] = ImageData.get_train_data()
        print("training data ready!")

        l_ll = [0.0] * len(l_weights)
        for i in range(len(l_weights)):
            l_ll[i] = []

        for e in range(100):
            for i in range(287):
                idx = i * 10
                for j in range(len(networks)):
                    loss = networks[j].train(x_train[idx:idx + 10, :], y_train[idx:idx + 10, :])
                    l_ll[j].append(loss)
            print("epoch", e)
            for i in range(len(l_weights)):
                hit_rate = get_hit_rate(networks[i])
                print("hit rate =", hit_rate, " with weight =", l_weights[i], "bias =", bias)
        self.l_ll = l_ll

    def show_result(self):
        l_weights = self.l_weights
        l_ll = self.l_ll
        plt.title('loss with different weight')
        for i in range(len(l_weights)):
            x = np.arange(1, len(l_ll[i]) + 1)
            label_str = 'weight = ' + str(l_weights[i])
            plt.plot(x[100:], l_ll[i][100:], label=(label_str))
        plt.legend()
        plt.xlabel('iteration times')
        plt.ylabel('loss')
        plt.show()

    def get_best_network(self):
        best_network = NetworkUtil.get_best_network(self.networks)
        return best_network
