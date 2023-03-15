import math

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


def save_val_prediction(image: np.array, mask: np.array, prediction: np.array, cmap: np.array, save_path: str) -> None:
    assert image.shape[:-1] == mask.shape == prediction.shape, f'验证集 image({image.shape[-2:]}), mask({mask.shape}), ' \
                                                               f'predict({prediction.shape}) 形状大小不一致'
    result = np.hstack([image, cmap[mask], cmap[prediction]])
    cv2.imwrite(save_path, result)


def save_test_prediction(image: np.array, prediction: np.array, cmap: np.array, save_path: str) -> None:
    assert image.shape[:-1] == prediction.shape, f'测试集 image({image.shape[-2:]}), ' \
                                                               f'predict({prediction.shape}) 形状大小不一致'
    result = cv2.addWeighted(image, 0.6, cmap[prediction], 0.4, 0, dtype=cv2.CV_32F).astype(np.uint8)
    cv2.imwrite(save_path, result)


class Record:
    def __init__(self, epoch_num: int, evaluation: list, save_path: str):
        self.evaluation = evaluation
        self.epoch_num = epoch_num
        self.save_path = save_path
        self.loss_list = []
        self.metric_dict_list = []

    def add_data_mean(self, loss: float, metric_dict: dict):

        assert sorted(metric_dict.keys()) == sorted(self.evaluation), f'传入验证结果和预定不一致'

        self.loss_list.append(loss)
        self.metric_dict_list.append(metric_dict)

    def draw(self):
        assert len(self.loss_list) == len(self.metric_dict_list), f'训练和验证次数不统一，无法记录训练过程'
        assert not len(self.loss_list) == 0, f'训练还未开始，无法记录训练过程'

        loss_max = math.ceil(self.loss_list[0])
        use_loss = [loss / loss_max for loss in self.loss_list]

        draw_dict = {}
        for el in self.evaluation:
            draw_dict[el] = [dc[el] for dc in self.metric_dict_list]

        x = [i + 1 for i in range(len(self.loss_list))]
        y_loss = use_loss

        plt.figure(figsize=(18, 8), dpi=80)
        plt.xticks(range(0, self.epoch_num + 1, self.epoch_num//20))
        plt.yticks([i/10 for i in range(1, 11)])

        plt.xlim(0, self.epoch_num)
        plt.ylim(0, 1)

        plt.xlabel('Epoch')
        plt.ylabel('Parameter')

        plt.grid(alpha=0.3, linestyle=':')

        plt.plot(x, y_loss, c='red', linestyle='-', label=f'Loss(*{loss_max:.2f})')
        for el in self.evaluation:
            plt.plot(x, draw_dict[el], linestyle='--', label=el)

        plt.legend(loc=0)
        plt.savefig(self.save_path)
