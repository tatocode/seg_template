import torch
import argparse
import datasets.temp
from datasets.temp import TempDataset
import albumentations as A
from torch.utils.data import DataLoader
from models.UNet.unet import UNet
from torch import nn
from tools.evaluate import SegmentationMetric
from tools.trainer import MyTrainer
from util.file import *
from util.functions import *
from tools.tester import MyTester

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_type', type=str, default='', help='Training or testing')
    parser.add_argument('--type_name', type=str, default='', help='Name the current training procedure')
    parser.add_argument('--data_root', type=str, default='', help='Specifies where the data set is stored')
    parser.add_argument('--num_classes', type=int, default=0, help='Number of classes to segmentation')

    parser.add_argument('--batch_size', type=int, default=4, help='Number of batch size')
    parser.add_argument('--num_worker', type=int, default=2, help='Number of threads reading data')
    parser.add_argument('--epoch_num', type=int, default=500, help='Number of epoch num')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Number of learning rate')
    parser.add_argument('--pre_trained', type=str, default='', help='Pre-train weight file location')

    return parser




if __name__ == '__main__':
    # parameter

    param = get_argparser().parse_args()

    assert param.task_type in ['train', 'test'], '请使用 --task_type 可选参数指定当前任务为 train 或者 test'

    if param.task_type == 'train':
        if param.type_name.strip() == '':
            train_name = 'train'
        else:
            train_name = param.type_name.strip()
        assert param.data_root != '', '请使用 --data_root 可选参数指定数据集位置'
        data_root = param.data_root
        batch_size = param.batch_size
        num_worker = param.num_worker
        epoch_num = param.epoch_num
        learning_rate = param.learning_rate
        assert param.num_classes != 0, '请使用 --num_classes 可选参数指定要分割的类别数'
        num_classes = param.num_classes
        pre_trained = param.pre_trained

        # process
        A_transform = A.Compose([
            A.Resize(128, 128),
            A.Flip(),
        ])

        train_dataset = TempDataset(data_root=data_root, transform=A_transform, dataset_type='train')
        val_dataset = TempDataset(data_root=data_root, transform=A_transform, dataset_type='val')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=False)

        model = UNet(3, num_classes)

        # 加载 pre_trained 权重
        if pre_trained != '':
            model.load_state_dict(torch.load(pre_trained))

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.65)
        criterion = nn.CrossEntropyLoss()
        metric = SegmentationMetric(num_classes)

        train_config_dict = {
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'model': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'epoch_num': epoch_num,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'criterion': criterion,
            'metric': metric,
        }

        trainer = MyTrainer(train_config_dict)

        home_dir = os.path.join('result', train_name, get_datetime_str())
        make_dir(home_dir)

        trainer.train(home_dir, datasets.temp.CMAP)

    else:
        if param.type_name.strip() == '':
            test_name = 'test'
        else:
            test_name = param.type_name.strip()
        assert param.data_root != '', '请使用 --data_root 可选参数指定数据集位置'
        data_root = param.data_root
        batch_size = param.batch_size
        num_worker = param.num_worker
        assert param.num_classes != 0, '请使用 --num_classes 可选参数指定要分割的类别数'
        num_classes = param.num_classes
        pre_trained = param.pre_trained.strip()

        test_dataset = TempDataset(data_root=data_root, transform=None, dataset_type='test')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, drop_last=False)

        model = UNet(3, num_classes)

        assert pre_trained != '', f'测试时需使用 --pre_trained 可选参数传入训练好的权重文件'
        # 加载 pre_trained 权重
        model.load_state_dict(torch.load(pre_trained))

        test_config_dict = {
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'model': model,
            'test_loader': test_loader,
        }

        tester = MyTester(test_config_dict)

        home_dir = os.path.join('result', test_name, get_datetime_str())
        make_dir(home_dir)

        tester.test(home_dir, datasets.temp.CMAP)


