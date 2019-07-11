from PIL import Image
import numpy as np


class ImageData:
    def __init__(self):
        pass

    @staticmethod
    def get_image_by_path(path, tag_num):
        im = Image.open(path)
        im_array = (np.array(im)).reshape(1,784)
        im_array = im_array*1.0
        tag_array = np.array([[tag_num*1.0]])
        im_array = np.concatenate((im_array,tag_array),axis=1)
        return im_array

    @staticmethod
    def transform_im2data():
        # 80% for training
        # 20% for testing
        total = 256
        train_length = int(256 * 0.8)
        test_length = total - train_length
        ImageData._save_data("train", 0, train_length)
        ImageData._save_data("test", train_length, test_length)

    # 这个方法用于保存转换之后的数据
    # use_type可以是train或者test
    # idx_begin 是起始图片，而length指明了长度
    @staticmethod
    def _save_data(use_type, idx_begin, length):
        prefix = "..\\TRAIN\\"
        middle = "\\"
        suffix = ".bmp"
        get_image_by_path = ImageData.get_image_by_path
        data = get_image_by_path(prefix + '1\\0.bmp', 10)
        # 192 x 14 image
        for tag_num in range(1, 15):
            for image_num in range(idx_begin, idx_begin + length):
                path = prefix + str(tag_num) + middle + str(image_num) + suffix
                im_array = get_image_by_path(path, tag_num)
                data = np.concatenate((data, im_array), axis=0)
        data = data[1:]
        np.random.shuffle(data)
        x = data[:, 0:-1]
        y_data = (data[:, -1]).reshape(14*length, 1)
        y = np.zeros([14*length, 14])
        for i in range(y_data.shape[0]):
            j = int(y_data[i][0]) - 1
            y[i][j] = 1.0
        save_path = "..\\DATA_TRANSFORMED\\"
        np.save(save_path + "x_" + use_type + ".npy", x)
        np.save(save_path + "y_" + use_type + ".npy", y)

    @staticmethod
    def get_train_data():
        path = "..\\DATA_TRANSFORMED\\"
        x_train = np.load(path + "x_train.npy")
        y_train = np.load(path + "y_train.npy")
        return [x_train,y_train]

    @staticmethod
    def get_test_data():
        path = "..\\DATA_TRANSFORMED\\"
        x_test = np.load(path + "x_test.npy")
        y_test = np.load(path + "y_test.npy")
        return [x_test,y_test]
