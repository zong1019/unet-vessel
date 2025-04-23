# -*-coding:utf-8 -*-

"""
#-------------------------------
# @Author : 肆十二
# @QQ : 3045834499 可定制毕设
#-------------------------------
# @File : perdict.py
# @Description: 模型批量预测
# @Software : PyCharm
# @Time : 2024/2/14 10:48
#-------------------------------
"""

import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet
import warnings

warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True")


# todo 需要封装为函数
def predict():
    # 选择设备，有cuda用cuda，没有就用cpu
    save_dir = os.path.expanduser('~/models/segment/unet_42-drive/images/sckl1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 确保保存目录存在
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except Exception as e:
            print(f"Error creating directory {save_dir}: {e}")
            return


    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)

    # 加载模型参数
    try:
        net.load_state_dict(torch.load(os.path.expanduser('HRF/sckl1.pth'), map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 测试模式
    net.eval()
    # 读取所有图片路径
    test_images_dir = os.path.expanduser('~/models/segment/unet_42-drive/data/HRF/Test_Images')
    tests_path = glob.glob(os.path.join(test_images_dir, '*.png'))
    if not tests_path:
        print(f"未找到任何测试图片，请检查路径是否正确：{test_images_dir}")
        return

    print(f"Found {len(tests_path)} images for prediction.")

    # 遍历素有图片
    for i, test_path in enumerate(tests_path):
        # 保存结果地址
        save_res_path = os.path.join(save_dir, os.path.basename(test_path))
        # 读取图片
        print(f"读取图片: {test_path}")
        img = cv2.imread(test_path)

        if img is None:
            print(f"无法读取图片: {test_path}")
            continue

        origin_shape = img.shape
        print(f"图片原始尺寸: {origin_shape}")
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (512, 512))
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(save_res_path, pred)
        print("{}: {}的预测结果已经保存在{}".format(i+1, test_path, save_res_path))

if __name__ == "__main__":
    predict()