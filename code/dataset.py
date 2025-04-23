import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import numpy as np

class Dataset_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'Training_Images/*.png')) # todo 更换数据集原始图像读取逻辑，根据后缀进行读取
        # 添加调试信息
        print(f"数据路径: {data_path}")
        print(f"找到的图片文件数: {len(self.imgs_path)}")

        if len(self.imgs_path) == 0:
            raise ValueError(f"No images found in the specified path: {self.data_path}")

    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('Training_Images', 'Training_Labels')
        # label_path = label_path.replace('.png', '_manual1.png')  # todo 更新标签文件的逻辑
        label_path = label_path.replace('.png', '.png')  # todo 更新标签文件的逻辑

        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # image = cv2.imread(image_path)
        # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)c

        # 添加调试信息
        if image is None:
            print(f"Error: Could not read image file {image_path}")
        if label is None:
            print(f"Error: Could not read label file {label_path}")
        assert label is not None, f"Error: Label image {label_path} is empty."
        image = cv2.resize(image, (512, 512))
        label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


if __name__ == "__main__":

    data_path = "/home/x4228/models/segment/unet_42-drive/data/CHASEDB1"
    dataset = Dataset_Loader(data_path)
    print("数据个数：", len(dataset))
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=2,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
