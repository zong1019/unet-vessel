
import os
from model.unet_model import UNet
from utils.dataset import Dataset_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops import rearrange
import torch.nn.functional as F
import torch.optim as optim
# 定义 Dice Loss 函数
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.clamp(pred, smooth, 1.0 - smooth)  # 避免全零或全一
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


# 定义 KL 散度损失函数
def kl_divergence_loss(pred, target):
    pred = torch.softmax(pred, dim=1)
    target = torch.softmax(target, dim=1)
    loss = F.kl_div(pred.log(), target, reduction='batchmean')
    return loss

# 定义 DKl Loss 函数，将 Dice Loss 和 KL 散度损失函数加权组合
class KLAndDiceLoss(nn.Module):
    def __init__(self, kl_weight=0.5, dice_weight=0.5):
        super(KLAndDiceLoss, self).__init__()
        self.kl_weight = kl_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        # KL Divergence Loss
        kl_loss = kl_divergence_loss(inputs, targets)

        # Dice Loss
        dice_loss_value = dice_loss(inputs, targets)

        # Combined Loss
        combined_loss = self.kl_weight * kl_loss + self.dice_weight * dice_loss_value

        return combined_loss


def train_net(net, device, data_path, epochs=100, batch_size=1, lr=0.00001):
    # 加载数据集
    dataset = Dataset_Loader(data_path)
    per_epoch_num = len(dataset) / batch_size
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 定义优化器
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-08, amsgrad=False)

    criterion = KLAndDiceLoss(kl_weight=1, dice_weight=0.5)
    # 最优损失初始化
    best_loss = float('inf')
    # 开始训练
    loss_record = []
    with tqdm(total=epochs * per_epoch_num) as pbar:
        for epoch in range(epochs):
            net.train()
            for image, label in train_loader:
                image = image / 255.0
                if torch.isnan(image).any() or torch.isinf(image).any() or torch.isnan(label).any() or torch.isinf(
                        label).any():
                    print("Data contains NaN or Inf values")
                    continue
                optimizer.zero_grad()

                # 将数据转换为浮点数类型
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)

                # 前向传播
                pred = net(image)

                loss = criterion(pred, label)
                pbar.set_description("Processing Epoch: {} Loss: {}".format(epoch + 1, loss))

                # 检查损失值是否有效
                if not torch.isnan(loss) and not torch.isinf(loss):
                    # 如果当前的损失比最好的损失小，则保存当前轮次的模型
                    if loss < best_loss:
                        best_loss = loss
                        torch.save(net.state_dict(), 'unet.pth')  # 保存模型

                loss.backward()
                optimizer.step()
                pbar.update(1)

                loss_record.append(loss.item())

    # 绘制损失折线图
    plt.figure()
    plt.plot([i + 1 for i in range(0, len(loss_record))], loss_record)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('Test-results/training_loss.png')
def load_pretrained_weights(model, pretrained_path):

    pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    # 过滤不匹配的层权重
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    # 更新现有的model_dict
    model_dict.update(pretrained_dict)
    # 加载我们真正需要的state_dict
    model.load_state_dict(model_dict)
    return model

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet(n_channels=3, n_classes=1)

    net.to(device=device)
    pretrained_path = 'unet.pth'
    if os.path.exists(pretrained_path):
        print("加载预训练权重...")
        net = load_pretrained_weights(net, pretrained_path)
        print("预训练权重加载完毕！")

    # 指定训练集地址，开始训练
    data_path = "/home/x4228/vessel/unet_42-drive/data/DRIVE-SEG-DATA"
    print("进度条出现卡着不动不是程序问题，是他正在计算，请耐心等待")
    time.sleep(1)
    train_net(net, device, data_path, epochs=100,  batch_size=1)  # 开始训练，如果你GPU的显存小于4G，这里只能使用CPU来进行训练。

