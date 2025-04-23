# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: unet
File Name: utils_metrics.py
Author: chenming
Create Date: 2022/2/7
Description：
-------------------------------------------------
"""
import csv
import os
from os.path import join
os.makedirs("Test-results", exist_ok=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib
#matplotlib.use('TKAgg') #修改 matplotlib 后端为 TkAgg
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns



def f_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice系数
    # --------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score


# 设标签宽W，长H
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes):
    print('Num classes', num_classes)
    # -----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    # -----------------------------------------#
    hist = np.zeros((num_classes, num_classes))

    # ------------------------------------------------#
    #   获得验证集标签路径列表，方便直接读取
    #   获得验证集图像分割结果路径列表，方便直接读取
    # ------------------------------------------------#
    gt_imgs = [join(gt_dir, x + "_manual1.png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + "_manual1.png") for x in png_name_list]

    # ------------------------------------------------#
    #   读取每一个（图片-标签）对
    # ------------------------------------------------#
    for ind in range(len(gt_imgs)):
        # ------------------------------------------------#
        #   读取一张图像分割结果，转化成numpy数组
        # ------------------------------------------------#
        # todo 用于伪彩色图像
        # pred = np.array(Image.open(pred_imgs[ind]))
        # label = np.array(Image.open(gt_imgs[ind]))
        # ------------------------------------------------#
        #   读取一张对应的标签，转化成numpy数组
        # ------------------------------------------------#
        pred = np.array(cv2.imread(pred_imgs[ind]))
        label = np.array(cv2.imread(gt_imgs[ind]))
        # print(pred_imgs[ind])
        # print(gt_imgs[ind])

        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        # ------------------------------------------------#
        #   对一张图片计算21×21的hist矩阵，并累加
        # ------------------------------------------------#
        label = np.array([int(x) for x in label.flatten()])
        # label[label == 255] = 1

        pred = np.array([int(x) for x in pred.flatten()])
        pred[pred == 255] = 1

        # pred = np.array([int(x) for x in pred.flatten()])
        # hist += fast_hist(label.flatten(), pred, num_classes)
        hist += fast_hist(label, pred, num_classes)
        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                ind,
                len(gt_imgs),
                100 * np.nanmean(per_class_iu(hist)),
                100 * np.nanmean(per_class_PA_Recall(hist)),
                100 * per_Accuracy(hist)
            )
            )
    # ------------------------------------------------#
    #   计算所有验证集图片的逐类别mIoU值
    # ------------------------------------------------#
    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)
    # ------------------------------------------------#
    #   逐类别输出一下mIoU值
    # ------------------------------------------------#
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
              + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2)) + '; Precision-' + str(
            round(Precision[ind_class] * 100, 2)))

    # -----------------------------------------------------------------#
    #   在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    # -----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(
        round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))
    return np.array(hist, int), IoUs, PA_Recall, Precision


def compute_mIoU_gray(gt_dir, pred_dir, png_name_list, num_classes, name_classes):
    os.environ[
        'QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/x4228/anaconda3/envs/unet/lib/python3.10/site-packages/cv2/qt/plugins'
    print('Num classes', num_classes)
    # -----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    # -----------------------------------------#
    hist = np.zeros((num_classes, num_classes))

    # ------------------------------------------------#
    #   获得验证集标签路径列表，方便直接读取
    #   获得验证集图像分割结果路径列表，方便直接读取
    # ------------------------------------------------#
    # gt_imgs = [join(gt_dir, x + "_manual1.png") for x in png_name_list]
    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    # ------------------------------------------------#
    #   读取每一个（图片-标签）对
    # ------------------------------------------------#
    for ind in range(len(gt_imgs)):
        # ------------------------------------------------#
        #   读取一张图像分割结果，转化成numpy数组
        # ------------------------------------------------#
        # todo 用于伪彩色图像
        # pred = np.array(Image.open(pred_imgs[ind]))
        # label = np.array(Image.open(gt_imgs[ind]))
        # ------------------------------------------------#
        #   读取一张对应的标签，转化成numpy数组
        # ------------------------------------------------#
        pred = np.array(cv2.imread(pred_imgs[ind]))
        label = np.array(cv2.imread(gt_imgs[ind]))
        # print(pred_imgs[ind])
        # print(gt_imgs[ind])

        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue
        # 显示图像和标签
        plt.subplot(1, 2, 1)
        plt.title('Prediction')
        plt.imshow(pred)

        plt.subplot(1, 2, 2)
        plt.title('Label')
        plt.imshow(label)

        #plt.show()
        plt.savefig("plot.png") #不需要立即显示图像，而是可以保存图像到文件再查看
        # ------------------------------------------------#
        #   对一张图片计算21×21的hist矩阵，并累加
        # ------------------------------------------------#
        label = np.array([int(x) for x in label.flatten()])
        label[label == 255] = 1

        pred = np.array([int(x) for x in pred.flatten()])
        pred[pred == 255] = 1

        hist += fast_hist(label, pred, num_classes)
        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                ind,
                len(gt_imgs),
                100 * np.nanmean(per_class_iu(hist)),
                100 * np.nanmean(per_class_PA_Recall(hist)),
                100 * per_Accuracy(hist)
            )
            )
    # ------------------------------------------------#
    #   计算所有验证集图片的逐类别mIoU值
    # ------------------------------------------------#
    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)
    # ------------------------------------------------#
    #   逐类别输出一下mIoU值
    # ------------------------------------------------#
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
              + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2)) + '; Precision-' + str(
            round(Precision[ind_class] * 100, 2)))

    # -----------------------------------------------------------------#
    #   在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    # -----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(
        round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))
    return np.array(hist, int), IoUs, PA_Recall, Precision


def adjust_shape(pred, target):
    """
    调整预测结果的形状以匹配真实标签的形状
    """
    pred_resized = np.zeros_like(target)
    pred_resized[:pred.shape[0], :pred.shape[1]] = pred
    return pred_resized


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
    plt.clf()  # 清理绘图区域
    fig = plt.gcf()
    axes = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val)
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values) - 1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()


def compute_f1_score(confusion_matrix):
    """
    通过混淆矩阵计算F1分数

    参数:
    confusion_matrix (list of list): 混淆矩阵，例如[[TN, FP], [FN, TP]]

    返回:
    float: F1分数
    """
    TN, FP, FN, TP = confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0], confusion_matrix[1][1]

    # 计算Precision和Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # 计算F1分数
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score

def compute_dice_coefficient(confusion_matrix):
    """
    通过混淆矩阵计算Dice分数

    参数:
    confusion_matrix (list of list): 混淆矩阵，例如[[TN, FP], [FN, TP]]

    返回:
    float: Dice分数
    """
    TN, FP, FN, TP = confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0], confusion_matrix[1][1]

    # 计算Dice分数
    dice_score = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

    return dice_score

#示例混淆矩阵
confusion_matrix = [[50, 10], [5, 35]]  # 假设有一个混淆矩阵
f1 = compute_f1_score(confusion_matrix)
dice = compute_dice_coefficient(confusion_matrix)
print(f"f1_score: {f1}")
print(f"dice_coefficient: {dice}")


def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size=12, f1_score=None, dice_coefficient=None,sensitivity=None, specificity=None):
    os.makedirs(miou_out_path, exist_ok=True)

    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs) * 100), "Intersection over Union",
                   os.path.join(miou_out_path, "mIoU.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Pixel Accuracy", \
                   os.path.join(miou_out_path, "mPA.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))

    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Recall", \
                   os.path.join(miou_out_path, "Recall.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision) * 100), "Precision", \
                   os.path.join(miou_out_path, "Precision.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    # 绘制混淆矩阵并保存为 PNG 图片
    plt.figure(figsize=(10, 8))
    sns.heatmap(hist, annot=True, fmt="d", cmap="Blues", xticklabels=name_classes, yticklabels=name_classes)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(miou_out_path, 'confusion_matrix.png'))
    plt.close()
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.png"))

    # 绘制标准化后的混淆矩阵
    hist_normalized = hist.astype('float') / hist.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(hist_normalized, cmap='Blues')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(name_classes)))
    ax.set_yticks(np.arange(len(name_classes)))
    ax.set_xticklabels(name_classes, rotation=90)
    ax.set_yticklabels(name_classes)
    for i in range(len(name_classes)):
        for j in range(len(name_classes)):
            ax.text(j, i, f'{hist_normalized[i, j]:.2f}', va='center', ha='center', color='black')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(os.path.join(miou_out_path, 'confusion_matrix_normalized.png'))
    plt.close()
    print("Save normalized confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix_normalized.png"))

    # 如果有 F1 Score 和 Dice 系数，则绘制柱状图
    if f1_score is not None and dice_coefficient is not None:
        # 创建 Matplotlib 图表
        plt.figure(figsize=(8, 6))
        # 绘制 F1 Score 和 Dice 系数的柱状图
        labels = ['F1 Score', 'Dice Coefficient']
        scores = [f1_score, dice_coefficient]
        x_pos = np.arange(len(labels))  # 生成连续的数字作为 x 轴坐标
        bars = plt.bar(x_pos, scores, color=['blue', 'green'])
        # 添加标题和标签
        plt.title('Model Performance')
        plt.ylabel('Scores')
        # 设置 x 轴刻度标签
        plt.xticks(x_pos, labels)
        # 在柱状图上添加数值标注
        for bar, score in zip(bars, scores):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{score:.2f}', va='bottom', ha='center', color='black',
                     fontsize=12)

        # 保存图表为图片文件
        plt.savefig(os.path.join(miou_out_path, 'performance_scores.png'))
        # 显示图表
        plt.show()
        print("Save performance_scores out to " + os.path.join(miou_out_path, "performance_scores.png"))
