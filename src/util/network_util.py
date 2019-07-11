import numpy as np
from util.image_data import ImageData


class NetworkUtil:
    def __init__(self):
        pass

    # hit rate 表示命中率
    @staticmethod
    def get_hit_rate(network):
        [x_test, y_test] = ImageData.get_test_data()
        y_predict = network.predict(x_test)
        y_maximum = (np.max(y_predict,axis=1)).reshape(y_predict.shape[0],1)
        y_predict = y_predict == y_maximum
        result = y_predict != y_test
        miss = np.sum(result) / 2
        miss_rate = miss / y_predict.shape[0]
        return 1-miss_rate

    ## 陷入循环，手动测试
    @staticmethod
    def one_by_one_test(best_network):
        l_char = ["苟","利","国","家","生","死","以","岂","因","祸","福","避","趋","之"]
        get_image_by_path = ImageData.get_image_by_path
        while True:
            tag = int(input("选择第几个字?（输入1到14数字）\n"))
            number = int(input("第几张图片?（输入0到255数字）\n"))
            path = "..\\TRAIN\\" + str(tag) + "\\" + str(number) + ".bmp"
            im_array = get_image_by_path(path, tag)
            predict = best_network.predict(im_array)
            for i in range(predict.shape[1]):
                print(l_char[i],'的概率是：',predict[0][i])

    ## 挑出最佳的神经网络
    @staticmethod
    def get_best_network(networks):
        print('将会挑出'+str(len(networks))+'中最好的一个！')
        max_hit = 0.0
        best_network = networks[0]
        for i in range(len(networks)):
            hit_rate = NetworkUtil.get_hit_rate(networks[i])
            if hit_rate > max_hit:
                max_hit = hit_rate
                best_network = networks[i]
        print('最佳命中率是',max_hit)
        return best_network
