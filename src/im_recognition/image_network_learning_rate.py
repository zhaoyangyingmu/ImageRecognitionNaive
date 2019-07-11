import numpy as np
import matplotlib.pyplot as plt
from util.network_elements import Network
from util.image_data import ImageData
from util.network_util import NetworkUtil


class CompareLearningRate:
    def __init__(self):
        # set up network
        net_config = [784, 14, -1]
        l_learning_rate = [0.01, 0.03, 0.1, 0.3]
        weight = -0.10
        bias = 0.5
        networks = []
        for learning_rate in l_learning_rate:
            networks.append(Network(net_config, learning_rate, weight, bias))
        print("networks ready!")
        self.l_learning_rate = l_learning_rate
        self.networks = networks
        self.l_ll = []
        pass

    def training(self):
        l_learning_rate = self.l_learning_rate
        networks = self.networks
        get_hit_rate = NetworkUtil.get_hit_rate
        # training
        [x_train, y_train] = ImageData.get_train_data()
        print("training data ready!")

        l_ll = [0.0] * len(l_learning_rate)
        for i in range(len(l_learning_rate)):
            l_ll[i] = []

        for e in range(100):
            for i in range(287):
                idx = i * 10
                for j in range(len(networks)):
                    loss = networks[j].train(x_train[idx:idx + 10, :], y_train[idx:idx + 10, :])
                    l_ll[j].append(loss)
            print("epoch", e)
            for i in range(len(l_learning_rate)):
                hit_rate = get_hit_rate(networks[i])
                print("hit rate =", hit_rate, " with learning rate =", l_learning_rate[i])
        self.l_ll = l_ll

    def show_result(self):
        l_learning_rate = self.l_learning_rate
        l_ll = self.l_ll
        plt.title('loss with different learning rate')
        for i in range(len(l_learning_rate)):
            x = np.arange(1, len(l_ll[i]) + 1)
            label_str = 'leaning rate = ' + str(l_learning_rate[i])
            plt.plot(x[100:], l_ll[i][100:], label=(label_str))
        plt.legend()
        plt.xlabel('iteration times')
        plt.ylabel('loss')
        plt.show()

    def get_best_network(self):
        best_network = NetworkUtil.get_best_network(self.networks)
        return best_network
