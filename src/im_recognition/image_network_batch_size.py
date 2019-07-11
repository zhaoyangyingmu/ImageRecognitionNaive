import numpy as np
import matplotlib.pyplot as plt
from util.network_elements import Network
from util.image_data import ImageData
from util.network_util import NetworkUtil


class CompareBatchSize:
    def __init__(self):
        # set up network
        net_config = [784, 14, -1]
        learning_rate = 0.1
        weight = -0.1
        bias = 0.5
        l_batch_size = [1, 5, 10, 14, 28, 2870]
        networks = []
        for i in range(len(l_batch_size)):
            networks.append(Network(net_config, learning_rate, weight, bias))
        print("networks ready!")
        self.l_batch_size = l_batch_size
        self.networks = networks
        self.l_ll = []
        pass

    def training(self):
        l_batch_size = self.l_batch_size
        networks = self.networks
        get_hit_rate = NetworkUtil.get_hit_rate
        # training
        [x_train, y_train] = ImageData.get_train_data()
        print("training data ready!")

        l_ll = [0.0] * len(l_batch_size)
        for i in range(len(l_batch_size)):
            l_ll[i] = []
        for e in range(100):
            for j in range(len(l_batch_size)):
                for i in range(int(2870 / l_batch_size[j])):
                    idx = i * l_batch_size[j]
                    loss = networks[j].train(x_train[idx:idx + l_batch_size[j], :],
                                             y_train[idx:idx + l_batch_size[j], :])
                    l_ll[j].append(loss)
            print("epoch", e)
            for i in range(len(l_batch_size)):
                hit_rate = get_hit_rate(networks[i])
                print("hit rate =", hit_rate, " with batch size =", l_batch_size[i])
        self.l_ll = l_ll

    def show_result(self):
        l_batch_size = self.l_batch_size
        l_ll = self.l_ll
        plt.title('loss with different batch size')
        for i in range(len(l_batch_size)):
            x = np.arange(1, len(l_ll[i]) + 1)
            label_str = 'batch size = ' + str(l_batch_size[i])
            plt.plot(x[100:], l_ll[i][100:], label=(label_str))
        plt.legend()
        plt.xlabel('iteration times')
        plt.ylabel('loss')
        plt.show()

    def get_best_network(self):
        best_network = NetworkUtil.get_best_network(self.networks)
        return best_network


