from util.image_data import ImageData
from im_recognition.image_network_batch_size import CompareBatchSize
from im_recognition.image_network_bias import CompareBias
from im_recognition.image_network_learning_rate import CompareLearningRate
from im_recognition.image_network_weight import CompareWeight

if __name__ == '__main__':
    # # 先将图片的数据转换为np数组，然后保存到DATA_TRANSFORMED文件夹
    # ImageData.transform_im2data()
    #
    # # 比较不同的batch size
    # compare = CompareBatchSize()
    # compare.training()
    # compare.show_result()

    # # 比较不同的bias
    # compare = CompareBias()
    # compare.training()
    # compare.show_result()

    # # 比较不同的bias
    # compare = CompareLearningRate()
    # compare.training()
    # compare.show_result()

    # 比较不同的weight
    compare = CompareWeight()
    compare.training()
    compare.show_result()
