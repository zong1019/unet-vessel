import time
from tqdm import tqdm
from utils.utils_metrics import (compute_mIoU_gray, show_results, compute_dice_coefficient, compute_f1_score)
                                 #get_f1_score, get_dice_score, get_sensitivity, get_specificity)
import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet
import matplotlib.pyplot as plt
# 设置matplotlib为无头模式
import matplotlib
matplotlib.use('Agg')

def draw_plot_func(values, name_classes, title, ylabel, save_path, tick_font_size=12, plt_show=True):
    fig = plt.figure()
    plt.bar(range(len(values)), values, tick_label=name_classes)
    plt.title(title, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(rotation=90, fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    if plt_show:
        plt.show()
    plt.close(fig)


def calculate_confusion_matrix(preds, targets):

    TP = np.sum((preds == 255) & (targets == 255))
    FP = np.sum((preds == 255) & (targets == 0))
    FN = np.sum((preds == 0) & (targets == 255))
    TN = np.sum((preds == 0) & (targets == 0))

    return [[TN, FP], [FN, TP]]
def cal_miou(test_dir="~/vessel/unet_42-drive/data/DRIVE-SEG-DATA/Test_Images",
             pred_dir="~/vessel/unet_42-drive/data/DRIVE-SEG-DATA/results",
             gt_dir="~/vessel/unet_42-drive/data/DRIVE-SEG-DATA/Test_Labels",
             model_path='Error testing1113.pth'):
    name_classes = ["background", "vein"]
    num_classes = len(name_classes)

    test_dir = os.path.expanduser(test_dir)
    pred_dir = os.path.expanduser(pred_dir)
    gt_dir = os.path.expanduser(gt_dir)
    model_path = os.path.expanduser(model_path)

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    print("---------------------------------------------------------------------------------------")
    print("加载训练好的模型,模型位于{}".format(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    print("模型加载成功！")

    img_names = os.listdir(test_dir)
    image_ids = [image_name.split(".")[0] for image_name in img_names]

    print("---------------------------------------------------------------------------------------")
    print("对测试集进行批量推理")
    time.sleep(1)

    all_preds = []
    all_targets = []

    for image_id in tqdm(image_ids):
        image_path = os.path.join(test_dir, image_id + ".png")
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告：无法读取图像文件 {image_path}")
            continue
        origin_shape = img.shape

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (512, 512))
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        img_tensor = torch.from_numpy(img).to(device=device, dtype=torch.float32)
        pred = net(img_tensor)
        pred = np.array(pred.data.cpu()[0])[0]
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        pred_resized = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), pred_resized)

        gt_path = os.path.join(gt_dir, image_id + ".png")
        target = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if target is None:
            print(f"警告：无法读取标签文件 {gt_path}")
            continue

        all_preds.append(pred_resized)
        all_targets.append(target)

    print("测试集批量推理结束")
    print("开始计算MIOU等测试指标")

    hist, IoUs, PA_Recall, Precision = compute_mIoU_gray(gt_dir, pred_dir, image_ids, num_classes, name_classes)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)


    # 计算混淆矩阵
    confusion_matrix = calculate_confusion_matrix(all_preds, all_targets)

    # # 计算F1分数
    # f1_score = get_f1_score(pred, target)
    f1_score = compute_f1_score(confusion_matrix)

    dice_coefficient = compute_dice_coefficient(confusion_matrix)


    miou_out_path = "Test-results/Error testing/1113"
    if not os.path.exists(miou_out_path):
        os.makedirs(miou_out_path)

    show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size=12, f1_score=f1_score,dice_coefficient=dice_coefficient)
    #, sensitivity=sensitivity, specificity=specificity)
    print("测试指标计算成功，测试结果已经保存在DRIVE-SEG-DATA_Test-results目录下")


if __name__ == '__main__':
    cal_miou(test_dir="~/vessel/unet_42-drive/data/DRIVE-SEG-DATA/Test_Images",
                 pred_dir="~/vessel/unet_42-drive/data/DRIVE-SEG-DATA/results",
                 gt_dir="~/vessel/unet_42-drive/data/DRIVE-SEG-DATA/Test_Labels",
                 model_path='unet.pth')
def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size=12, f1_score=None, dice_coefficient=None):
    os.makedirs(miou_out_path, exist_ok=True)

    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nan))