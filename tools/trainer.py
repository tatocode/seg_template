import tools.logging as logging
import os
import random

# import torchvision.util
# import draw
import torch
from util.file import *
from util.data_process import *
from util.draw import *
import torch.nn.functional as F


class MyTrainer:
    """
    train_config_dict:
        device: 使用 GPU 还是 CPU
        model: 网络结构
        train_loader: 训练数据 loader
        val_loader: 验证数据 loader
        epoch_num: 迭代次数
        optimizer: 优化器
        scheduler: 调度器
        criterion: 损失函数
        metric: 验证集上评价
    """

    def __init__(self, train_config_dict: dict):
        self.device = train_config_dict['device']
        self.model = train_config_dict['model'].to(self.device)
        self.train_loader = train_config_dict['train_loader']
        self.val_loader = train_config_dict['val_loader']
        self.epoch_num = train_config_dict['epoch_num']
        self.optimizer = train_config_dict['optimizer']
        self.scheduler = train_config_dict['scheduler']
        self.criterion = train_config_dict['criterion']
        self.metric = train_config_dict['metric']

    def train(self, home_dir, cmap, mean=None, std=None):
        if std is None:
            std = [0.229, 0.224, 0.225]
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        print(f'>>> {self.device} 可用!')

        checkpoint_dir = os.path.join(home_dir, 'checkpoint')
        save_val_dir = os.path.join(home_dir, 'save_val')

        make_dir(home_dir)
        make_dir(checkpoint_dir)
        make_dir(save_val_dir)

        logger = logging.getLogger(home_dir)

        record = Record(self.epoch_num, ['mAcc', 'mIoU', 'FWIoU'], os.path.join(home_dir, r'record.jpg'))

        # train
        mIoU_best = 0
        for epoch in range(self.epoch_num):
            self.model.train()

            loss_epoch = 0
            time_interval = 0
            for image, mask in self.train_loader:

                image, mask = image.to(self.device, dtype=torch.float32), mask.to(self.device, dtype=torch.long)

                self.optimizer.zero_grad()
                prediction = self.model(image)
                loss = self.criterion(prediction, mask)
                loss.backward()
                self.optimizer.step()

                loss_np = loss.detach().cpu().numpy()
                loss_epoch += loss_np.item()
                time_interval += 1

            loss_mean = loss_epoch/time_interval
            logger.info(f'train | epoch: {epoch + 1}\t\t\t loss: {loss_mean:.4f}')

            self.model.eval()
            with torch.no_grad():
                mAcc_sum = 0
                mIoU_sum = 0
                FWIoU_sum = 0
                time_interval = 0
                for image, mask in self.val_loader:
                    image, mask = image.to(self.device, dtype=torch.float32), mask.to(self.device, dtype=torch.long)
                    prediction = torch.argmax(self.model(image).cpu(), dim=1)
                    self.metric.addBatch(prediction, mask.cpu())
                    mAcc_sum += self.metric.meanPixelAccuracy()
                    mIoU_sum += self.metric.meanIntersectionOverUnion()
                    FWIoU_sum += self.metric.Frequency_Weighted_Intersection_over_Union()
                    self.metric.reset()

                    # 画图
                    idx_random = random.randint(0, image.shape[0]-1)

                    i = de_normalization(image[idx_random].cpu(), mean=mean, std=std).numpy().transpose(1, 2, 0)
                    m = mask[idx_random].cpu().numpy()
                    p = prediction[idx_random].cpu().numpy()

                    save_val_prediction(i, m, p, cmap, os.path.join(save_val_dir, f'epoch_{epoch+1}.jpg'))
                    time_interval += 1

                mAcc_mean = mAcc_sum/time_interval
                mIoU_mean = mIoU_sum/time_interval
                FWIoU_mean = FWIoU_sum/time_interval

                logger.info(f' val  | epoch: {epoch + 1}\t\t\t mAcc: {mAcc_mean:.2f}, mIoU: {mIoU_mean:.2f}, FWIoU: {FWIoU_mean:.2f}')

                torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, 'last.pth'))
                if mIoU_mean > mIoU_best:
                    mIoU_best = mIoU_mean
                    torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))

            self.scheduler.step()

            # 记录训练过程并画图
            metric_dict = {
                'mAcc': mAcc_mean,
                'mIoU': mIoU_mean,
                'FWIoU': FWIoU_mean
            }
            record.add_data_mean(loss_mean, metric_dict)
            record.draw()

