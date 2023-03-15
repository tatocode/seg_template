import matplotlib.pyplot as plt
import numpy as np
import datetime

class DrawConfusionMatrix:
    def __init__(self, labels_name, normalize=True):
        """
		normalize：是否设元素为百分比形式
        """
        self.normalize = normalize
        self.labels_name = labels_name
        self.num_classes = len(labels_name)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")

    def update(self, predicts, labels):
        """

        :param predicts: 一维预测向量，eg：array([0,5,1,6,3,...],dtype=int64)
        :param labels:   一维标签向量：eg：array([0,5,0,6,2,...],dtype=int64)
        :return:
        """
        for predict, label in zip(predicts, labels):
            self.matrix[label, predict] += 1

    def getMatrix(self,normalize=True):
        """
        根据传入的normalize判断要进行percent的转换，
        如果normalize为True，则矩阵元素转换为百分比形式，
        如果normalize为False，则矩阵元素就为数量
        Returns:返回一个以百分比或者数量为元素的矩阵

        """
        if normalize:
            per_sum = self.matrix.sum(axis=1)  # 计算每行的和，用于百分比计算
            for i in range(self.num_classes):
                self.matrix[i] =(self.matrix[i] / per_sum[i])   # 百分比转换
            self.matrix=np.around(self.matrix, 2)   # 保留2位小数点
            self.matrix[np.isnan(self.matrix)] = 0  # 可能存在NaN，将其设为0
        return self.matrix

    def drawMatrix(self, file_name):
        self.matrix = self.getMatrix(self.normalize)
        plt.imshow(self.matrix, cmap=plt.cm.Blues)  # 仅画出颜色格子，没有值
        plt.title("Normalized confusion matrix")  # title
        plt.xlabel("Predict label")
        plt.ylabel("Truth label")
        plt.yticks(range(self.num_classes), self.labels_name)  # y轴标签
        plt.xticks(range(self.num_classes), self.labels_name)  # x轴标签

        for x in range(self.num_classes):
            for y in range(self.num_classes):
                value = float(format('%.2f' % self.matrix[y, x]))  # 数值处理
                plt.text(x, y, value, verticalalignment='center', horizontalalignment='center')  # 写值

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

        plt.colorbar()  # 色条
        plt.savefig(file_name, bbox_inches='tight')  # bbox_inches='tight'可确保标签信息显示全
        plt.show()


def get_datetime_str(style='dt'):
    '''
    获取当前时间字符串
    :param style: 'dt':日期+时间；'date'：日期；'time'：时间
    :return: 当前时间字符串
    '''
    cur_time = datetime.datetime.now()

    date_str = cur_time.strftime('%y%m%d')
    time_str = cur_time.strftime('%H%M%S')

    if style == 'date':
        return date_str
    elif style == 'time':
        return time_str
    else:
        return date_str + '_' + time_str


# if __name__ == '__main__':
#     label_name = ['cat', 'dog']
#     d = DrawConfusionMatrix(labels_name=label_name, normalize=True)
#     iter = 100
#     for i in range(iter):
#         pred = np.random.random((10))
#         pred[pred>0.5]=1
#         pred[pred<0.5]=0
#         label = np.random.random((10))
#         label[label>0.5] = 1
#         label[label<0.5]=0
#         d.update(np.array(pred, dtype=np.int64), np.array(label, dtype=np.int64))
#     d.drawMatrix('a.png')

