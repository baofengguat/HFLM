import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from check_dataset import check_dataset
# from check_model import check_model
from l2t_ww.utils.utils import AverageMeter, accuracy, set_logging_config
# from train.meta_optimizers import MetaSGD

from focal_loss import FocalLoss
torch.backends.cudnn.benchmark = True

def _get_num_features(model):#根据输入的model名称来确定相应的特征数量（num_features）
    if model.startswith('resnet'):
        n = int(model[6:])
        if n in [18, 34]:
            return [64, 64, 128, 256, 512]
        elif n in [50,101,152]:
            return [64, 256, 512, 1024, 2048]
        else:
            n = (n-2) // 6
            return [16]*n+[32]*n+[64]*n
    elif model.startswith('vgg'):
        n = int(model[3:].split('_')[0])
        if n == 9:
            return [64, 128, 256, 512, 512]
        elif n == 16:#11--16
            return [64, 128, 256, 512, 512]
    elif model.startswith('UNet'):
        return [None,32,64,128,256]
    raise NotImplementedError


class FeatureMatching(nn.ModuleList):#目的是实现特征匹配功能，用于两个不同模型的特征之间的匹配
    def __init__(self, source_model, target_model, pairs):
        super(FeatureMatching, self).__init__()
        self.src_list = _get_num_features(source_model)#[0，1，2，3，4，5]
        self.tgt_list = _get_num_features(target_model)
        self.pairs = pairs

        for src_idx, tgt_idx in pairs:#取了[1,2,3,4]
            self.append(nn.Conv2d(self.tgt_list[tgt_idx], self.src_list[src_idx], 1))#使用了nn.Conv2d层，将目标模型的特征映射到源模型的特征维度

    def forward(self, source_features, target_features,weight, beta, loss_weight):
        matching_loss = 0.0#计算特征匹配损失
        for i, (src_idx, tgt_idx) in enumerate(self.pairs):
            sw = source_features[src_idx].size(3)
            tw = target_features[tgt_idx].size(3)
            # sconv_num=source_features[src_idx].size(1)
            # tconv_num=target_features[tgt_idx].size(1)
            if sw == tw :#and sconv_num==tconv_num:
                diff = source_features[src_idx] - self[i](target_features[tgt_idx])
            elif sw!=tw:#源特征和目标特征的尺寸不同，则使用F.interpolate()函数将目标特征调整为与源特征相同的尺寸
                diff = F.interpolate(
                    source_features[src_idx],
                    scale_factor=tw / sw,
                    mode='bilinear'
                ) - self[i](target_features[tgt_idx])
            # elif sconv_num!=tconv_num:
            #     diff = F.interpolate(
            #         source_features[src_idx],
            #         scale_factor=tconv_num / sconv_num,
            #         mode='bilinear'
            #     ) - self[i](target_features[tgt_idx])
            diff = diff.pow(2).mean(3).mean(2)#diff.pow(2)：对diff中的每个元素进行平方操作
            #.mean(3)：对diff的第三维（如果有）进行求和操作，即将第三维的元素转换为单个数值
            if loss_weight is None and weight is None:
                diff = diff.mean(1).mean(0).mul(beta[i])
            elif loss_weight is None:
                diff = diff.mul(weight[i]).sum(1).mean(0).mul(beta[i])
            elif weight is None:
                diff = (diff.sum(1)*(loss_weight[i].squeeze())).mean(0).mul(beta[i])
            else:
                diff = (diff.mul(weight[i]).sum(1)*(loss_weight[i].squeeze())).mean(0).mul(beta[i])
            matching_loss = matching_loss + diff
        return matching_loss


class WeightNetwork(nn.ModuleList):#目的是为输入的模型创建一个权重网络，用于对源模型的特征进行加权处理
    def __init__(self, source_model, pairs):
        super(WeightNetwork, self).__init__()
        n = _get_num_features(source_model)
        for i, _ in pairs:
            self.append(nn.Linear(n[i], n[i]))
            self[-1].weight.data.zero_()
            self[-1].bias.data.zero_()
        self.pairs = pairs

    def forward(self, source_features):
        outputs = []
        for i, (idx, _) in enumerate(self.pairs):
            f = source_features[idx]
            f = F.avg_pool2d(f, f.size(2)).view(-1, f.size(1))

            outputs.append(F.softmax(self[i](f), 1))

        return outputs


class LossWeightNetwork(nn.ModuleList):#目的是根据输入的参数来初始化一个权重网络，用于计算损失权重。
    def __init__(self, source_model, pairs, weight_type='relu', init=None):
        super(LossWeightNetwork, self).__init__()
        n = _get_num_features(source_model)
        if weight_type == 'const':
            self.weights = nn.Parameter(torch.zeros(len(pairs)))
        else:
            for i, _ in pairs:
                l = nn.Linear(n[i], 1)
                if init is not None:
                    nn.init.constant_(l.bias, init)
                self.append(l)
        self.pairs = pairs
        self.weight_type = weight_type

    def forward(self, source_features):
        outputs = []
        if self.weight_type == 'const':
            for w in F.softplus(self.weights.mul(10)):
                outputs.append(w.view(1, 1))
        else:
            for i, (idx, _) in enumerate(self.pairs):
                f = source_features[idx]
                f = F.avg_pool2d(f, f.size(2)).view(-1, f.size(1))
                if self.weight_type == 'relu':
                    outputs.append(F.relu(self[i](f)))
                elif self.weight_type == 'relu-avg':
                    outputs.append(F.relu(self[i](f.div(f.size(1)))))
                elif self.weight_type == 'relu6':
                    outputs.append(F.relu6(self[i](f)))
        return outputs

def inner_objective(data,opt=None,#定义了一个内部目标函数，用于在目标模型和源模型之间进行知识迁移
                    target_model=None,source_model=None,
                    wnet=None,lwnet=None,
                    target_branch=None,state=None,
                    logger=None,source_model_name=None,target_model_name=None,matching_only=False,
                    write_log=True,device="cuda:0"):
    x, y = data[0].to(device), data[1].to(device).long()
#函数的目的是计算一个损失函数，该损失函数是目标模型在给定数据上的预测与实际标签之间的差距，以及一个匹配损失，该损失是源模型和目标模型在给定数据上的特征之间的匹配程度
    y_pred, target_features = target_model.forward_with_features(x)

    with torch.no_grad():
        s_pred, source_features = source_model.forward_with_features(x)
#调用源模型的forward_with_features方法，该方法将输入数据传递给模型，并返回预测结果和特征。
    weights = wnet(source_features)#网络将源特征映射到损失权重
    state['loss_weights'] = ''
    if opt.loss_weight:#如果设置了损失权重，调用损失权重网络lwnet，该网络将源特征映射到损失权重
        loss_weights = lwnet(source_features)
        state['loss_weights'] = ' '.join(['{:.2f}'.format(lw.mean().item()) for lw in loss_weights])
    else:
        loss_weights = None
    beta = [opt.beta] * len(wnet)

    matching_loss = target_branch(source_features,
                                  target_features,
                                  weights, beta, loss_weights)

    state['accuracy'] = accuracy(y_pred.data, y, topk=(1,))[0].item()

    if matching_only:
        return matching_loss
    # loss_F = nn.CrossEntropyLoss()
    if opt.loss_func == "crossentropy":
        # criterion = nn.CrossEntropyLoss().cuda()
        loss = F.cross_entropy(y_pred, y)
    elif opt.loss_func == "focalloss":
        criterion = FocalLoss().cuda()
        loss = criterion(y_pred, y)

    state['loss'] = loss.item()
    if write_log:
        logger.info('[{} transfer to {}] [Epoch {:3d}] [Iter {:3d}] [Loss {:.4f}] [Acc {:.4f}] [LW {}]'.format(
            source_model_name, target_model_name,
            state['epoch'], state['iter'],
            state['loss'], state['accuracy'], state['loss_weights']))
    print('[{} transfer to {}] [Epoch {:3d}] [Iter {:3d}] [Loss {:.4f}] [Acc {:.4f}] [LW {}]'.format(source_model_name,
                                                                                                     target_model_name,
                                                                                                     state['epoch'],
                                                                                                     state['iter'],
                                                                                                     state['loss'],
                                                                                                     state['accuracy'],
                                                                                                     state[
                                                                                                         'loss_weights']))
    return loss + matching_loss

def outer_objective(data,opt=None,target_model=None,state=None,#这个函数主要用于训练神经网络模型，特别是用于分类任务
                    device="cuda:0"):
    x, y = data[0].to(device),data[1].to(device).long()
    y_pred, _ = target_model(x)
    state['accuracy'] = accuracy(y_pred.data, y, topk=(1,))[0].item()
    # loss_F = nn.CrossEntropyLoss()
    if opt.loss_func == "crossentropy":
        # criterion = nn.CrossEntropyLoss().cuda()
        loss = F.cross_entropy(y_pred, y)
    elif opt.loss_func == "focalloss":
        criterion = FocalLoss().cuda()
        loss = criterion(y_pred, y)
    state['loss'] = loss.item()
    return loss

def validate(model, loader,device="cuda:0"):#用于计算模型在给定数据加载器上的准确性，并在验证过程中跟踪和更新准确性
    acc = AverageMeter()
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_pred, _ = model(x)
        acc.update(accuracy(y_pred.data, y, topk=(1,))[0].item(), x.size(0))
    return acc.avg


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation

    def dice_coef(self, pred, gt):#dice_coef方法用于计算预测结果和真实标签之间的Dice系数
        softmax_pred = torch.nn.functional.softmax(pred, dim=1)
        seg_pred = torch.argmax(softmax_pred, dim=1)
        all_dice = 0
        gt = gt.squeeze(dim=1)
        batch_size = gt.shape[0]
        num_class = softmax_pred.shape[1]
        for i in range(num_class):
            each_pred = torch.zeros_like(seg_pred)
            each_pred[seg_pred == i] = 1

            each_gt = torch.zeros_like(gt)
            each_gt[gt == i] = 1

            intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)

            union = each_pred.view(batch_size, -1).sum(1) + each_gt.view(batch_size, -1).sum(1)
            dice = (2. * intersection) / (union + 1e-5)

            all_dice += torch.mean(dice)

        return all_dice * 1.0 / num_class

    def forward(self, pred, gt):#forward方法是模型的前向传播函数，它首先将预测结果进行softmax激活，然后将真实标签转换为类别标签
        sigmoid_pred = F.softmax(pred, dim=1)

        batch_size = gt.shape[0]
        num_class = sigmoid_pred.shape[1]

        bg = torch.zeros_like(gt)
        bg[gt == 0] = 1
        label1 = torch.zeros_like(gt)
        label1[gt == 1] = 1
        label2 = torch.zeros_like(gt)
        label2[gt == 2] = 1
        label = torch.cat([bg, label1, label2], dim=1)

        loss = 0
        smooth = 1e-5
        #计算每个类别的损失，最后对所有类别的损失求平均值作为最终的损失。
        #在计算损失时，使用了Dice系数作为衡量预测结果与真实标签之间的相似性。
        for i in range(num_class):
            intersect = torch.sum(sigmoid_pred[:, i, ...] * label[:, i, ...])
            z_sum = torch.sum(sigmoid_pred[:, i, ...])
            y_sum = torch.sum(label[:, i, ...])
            loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss * 1.0 / num_class
        return loss


class JointLoss(nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()#交叉熵损失
        self.dice = DiceLoss()#Dice损失

    def forward(self, pred, gt):#pred是模型的预测结果，gt是真实标签
        ce = self.ce(pred, gt.squeeze(axis=1).long())#gt.squeeze(axis=1)将标签从形状(batch_size, 1)压缩为(batch_size)，以便与预测结果进行比较
                                                     #long()将其转换为整数类型，以便与交叉熵损失进行比较。
        return (ce + self.dice(pred, gt)) / 2  #计算交叉熵损失（ce）和Dice损失（self.dice(pred, gt)）的和，然后将结果除以2，得到最终的损失值