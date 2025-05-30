import time

import numpy as np
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import copy
import datetime
import random
from config import get_args
from utils import *
# from weight_perturbation import WPOptim
from utils_Package.weight_perturbation import WPOptim
from lw2w import WeightNetwork,LossWeightNetwork,FeatureMatching,inner_objective,outer_objective,validate
from l2t_ww.train.meta_optimizers import MetaSGD
from l2t_ww.check_model import check_model
from N_data_dataloaders import recorded_multicenters_data_API_dataloader
from focal_loss import FocalLoss
#dataset:camelyon17/prostate/Nuclei/5_hospital_lung_nodules/4_gastric_centers/cifar100/cifar10/tinyimage-net/LIDC
#fed:fedlwt/fedavg/fedprox/moon/harmofl

each_save_interval=1
current_dir = os.path.dirname(os.path.abspath(__file__))#获取main.py所在的文件夹路径
weights_path=os.path.join(current_dir,r'l2t_ww\resnet18-5c106cde.pth')#输入resnet模型预训练参数路径
weights_path1=os.path.join(current_dir,r'l2t_ww\vgg16-397923af.pth')#输入vgg模型预训练参数路径


###=====================迁移===========================
def init_pairs(opt):#将字符串表示的配对（如 "1-2"）转换为元组表示的配对（如 (1, 2)）。
    pairs = []
    for pair in opt.pairs.split(','):
        pairs.append((int(pair.split('-')[0]),
                      int(pair.split('-')[1])))
    return pairs
def init_meta_model(opt,pairs,server_model,local_nets):#初始化元模型，包括权重网络、损失权重网络、权重参数、目标参数和目标分支等
    local_models_name = opt.sites.copy()
    server_model_name = [opt.sites[opt.global_center_idx]]
    local_models_name.pop(opt.global_center_idx)#删除全局中心索引所对应的元素

    wnets=dict()#存储权重网络
    lwnets=dict()#存储损失权重网络
    wlw_weight_params=dict()#存储权重参数
    target_params_dict=dict()#存储目标参数
    target_branch_dict=dict()#存储目标分支
    for net_i in local_models_name:#处理不同模型之间的信息传递和参数同步
        wnet = WeightNetwork(opt.source_model, pairs).to(device)
        weight_params = list(wnet.parameters())#获取wnet的参数列表（weight_params）
        if opt.loss_weight:
            lwnet = LossWeightNetwork(opt.source_model, pairs, opt.loss_weight_type, opt.loss_weight_init).to(device)
            weight_params = weight_params + list(lwnet.parameters())
        if opt.wnet_path is not None:
            ckpt = torch.load(opt.wnet_path)
            wnet.load_state_dict(ckpt['w'])
            if opt.loss_weight:
                lwnet.load_state_dict(ckpt['lw'])
        #在联邦学习中处理不同模型之间的信息传递和参数同步
        target_branch = FeatureMatching(opt.source_model,
                                        opt.target_model,
                                        pairs).to(device)
        local_target_params = list(local_nets[net_i].parameters()) + copy.deepcopy(list(target_branch.parameters()))
        wnets["%sto%s"%(server_model_name[0],net_i)] = copy.deepcopy(wnet)
        lwnets["%sto%s"%(server_model_name[0],net_i)]=copy.deepcopy(lwnet)
        wlw_weight_params["%sto%s"%(server_model_name[0],net_i)]=copy.deepcopy(weight_params)
        target_params_dict["%sto%s"%(server_model_name[0],net_i)]=local_target_params
        target_branch_dict["%sto%s"%(server_model_name[0],net_i)]=copy.deepcopy(target_branch)

    for net_i in local_models_name:  # 处理不同模型之间的信息传递和参数同步
        wnet = WeightNetwork(opt.target_model, pairs).to(device)
        weight_params = list(wnet.parameters())  # 获取wnet的参数列表（weight_params）
        if opt.loss_weight:
            lwnet = LossWeightNetwork(opt.target_model, pairs, opt.loss_weight_type, opt.loss_weight_init).to(
                device)
            weight_params = weight_params + list(lwnet.parameters())
        if opt.wnet_path is not None:
            ckpt = torch.load(opt.wnet_path)
            wnet.load_state_dict(ckpt['w'])
            if opt.loss_weight:
                lwnet.load_state_dict(ckpt['lw'])
        target_branch = FeatureMatching(opt.target_model,
                                        opt.source_model,
                                        pairs).to(device)
        server_target_params = list(server_model[server_model_name[0]].parameters()) + copy.deepcopy(
            list(target_branch.parameters()))
        wnets["%sto%s" % (net_i, server_model_name[0])] = copy.deepcopy(wnet)
        lwnets["%sto%s" % (net_i, server_model_name[0])] = copy.deepcopy(lwnet)
        wlw_weight_params["%sto%s" % (net_i, server_model_name[0])] = copy.deepcopy(weight_params)
        target_params_dict["%sto%s" % (net_i, server_model_name[0])] = server_target_params
        target_branch_dict["%sto%s" % (net_i, server_model_name[0])] = copy.deepcopy(target_branch)
    return wnets,lwnets,wlw_weight_params,target_params_dict,target_branch_dict#,source_optimizer
def optimizer_init(opt,wlw_weight_params,target_model_dict,target_params_dict,target_barnch_dict):
    source_optimizers=dict()#n_parties,2n个优化器
    target_optimizers=dict()
    for i,(k,weight_params) in enumerate(wlw_weight_params.items()):
        if opt.source_optimizer == 'sgd':
            source_optimizer = optim.SGD(weight_params, lr=opt.meta_lr, weight_decay=opt.meta_wd, momentum=opt.momentum,
                                         nesterov=opt.nesterov)
        elif opt.source_optimizer == 'adam':
            source_optimizer = optim.Adam(weight_params, lr=opt.meta_lr, weight_decay=opt.meta_wd)
        source_optimizers[k]=source_optimizer
        if opt.meta_lr == 0:
            target_optimizer = optim.SGD(target_params_dict[k], lr=opt.lr, momentum=opt.momentum, weight_decay=opt.wd)
        else:
            target_optimizer = MetaSGD(target_params_dict[k],
                                       [target_model_dict[k.split("to")[-1]], target_barnch_dict[k]],
                                       lr=opt.lr,
                                       momentum=opt.momentum,
                                       weight_decay=opt.wd, rollback=True, cpu=opt.T > 2)
        target_optimizers[k]=target_optimizer
    return source_optimizers,target_optimizers
#=====================================================================


#==========================联邦学习====================================
def init_nets(args, device='cpu',server=False):#初始化网络并加载预训练权重
    if args.alg=="fedlwt":
        local_models=args.sites.copy()
        server_model=[args.sites[args.global_center_idx]]
        local_models.pop(args.global_center_idx)
    else:
        local_models = args.sites.copy()
        server_model = ["server"]
    ##fine tune weights

    checkpoint = torch.load(weights_path, map_location=device)
    checkpoint1 = torch.load(weights_path1, map_location=device)

    #args.model=args.source_model if args.source_model== args.target_model else None####注意 如果args.source_model等于args.target_model，那么不会加载任何权重
    args.model = args.source_model
    if server:
        nets = {net_i: None for net_i in server_model}

        for net_i in server_model:
            server_net=check_model(args).to(device)
            ###finetune imagenet
            new_params = server_net.state_dict().copy()
            for name, param in new_params.items():
                # print(name)
                if name in checkpoint and param.size() == checkpoint[name].size():
                    new_params[name].copy_(checkpoint[name])
                    # print('copy {}'.format(name))
            server_net.load_state_dict(new_params)
            nets[net_i] = server_net
        model_meta_data = []
        layer_type = []
        for (k, v) in nets[net_i].state_dict().items():
            model_meta_data.append(v.shape)
            layer_type.append(k)

        return nets, model_meta_data, layer_type
    else:
        nets={net_i:None for net_i in local_models}
        args.model = "vgg16"
        for net_i in local_models:
            local_net = check_model(args).to(device)
            ###finetune imagenet
            new_params = local_net.state_dict().copy()
            for name, param in new_params.items():
                # print(name)
                if name in checkpoint1 and param.size() == checkpoint1[name].size():
                    new_params[name].copy_(checkpoint1[name])
                    # print('copy {}'.format(name))
            local_net.load_state_dict(new_params)

            nets[net_i] = local_net
        model_meta_data = []
        layer_type = []
        for (k, v) in nets[net_i].state_dict().items():
            model_meta_data.append(v.shape)
            layer_type.append(k)

        return nets, model_meta_data, layer_type
# sim_dict={}

#使用FedAvg算法训练神经网络，并在训练过程中计算准确性、混淆矩阵和AUC
def train_net(net_id, net, train_dataloader,val_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu",write_log=True):
    '''
    use for fedavg
    '''
    # net = nn.DataParallel(net)
    net.cuda()

    if write_log:
        logger.info('Training network %s' % str(net_id))
        logger.info('n_training: %d' % len(train_dataloader))
        logger.info('n_test: %d' % len(test_dataloader))
    print('Training network %s' % str(net_id))
    print('n_training: %d' % len(train_dataloader))
    print('n_test: %d' % len(test_dataloader))
    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:  # 有些数据没有验证集
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    else:
        val_acc, val_auc = None, None
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)

    logger.info('>> Pre-Training Training accuracy: {}--->>auc:{}'.format(train_acc, train_auc))

    logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    logger.info('>> Pre-Training Test accuracy: {}------>>auc:{}'.format(test_acc, test_auc))
    print('>> Pre-Training Training accuracy: {}-------->>auc:{}'.format(train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Pre-Training Test accuracy: {}------------>>auc:{}'.format(test_acc, test_auc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()#定义损失函数

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out,_ = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)
    if write_log:
        logger.info('>> Training accuracy: {}---->>auc:{}'.format(train_acc, train_auc))
        logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
        logger.info('>> Test accuracy: {}----->>auc:{}'.format(test_acc, test_auc))
        logger.info(' ** Training complete **')
    print('>> Training accuracy: %f------->>auc:%f' % (train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Test accuracy: %f----------->>auc:%f' % (test_acc, test_auc))
    net.to('cpu')

    print(' ** Training complete **')
    return train_acc, val_acc, test_acc, train_auc, val_auc, test_auc

def train_net_fedprox(net_id, net, global_net, train_dataloader,val_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      device="cpu",write_log=True):
    '''
    use for fedprox
    '''
    # global_net.to(device)
    # net = nn.DataParallel(net)
    net.cuda()
    # else:
    if write_log:
        logger.info('Training network %s' % str(net_id))
        logger.info('n_training: %d' % len(train_dataloader))
        logger.info('n_test: %d' % len(test_dataloader))
    print('Training network %s' % str(net_id))
    print('n_training: %d' % len(train_dataloader))
    print('n_test: %d' % len(test_dataloader))
    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:  # 有些数据没有验证集
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    else:
        val_acc, val_auc = None, None
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)

    logger.info('>> Pre-Training Training accuracy: {}--->>auc:{}'.format(train_acc, train_auc))

    logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    logger.info('>> Pre-Training Test accuracy: {}------>>auc:{}'.format(test_acc, test_auc))
    print('>> Pre-Training Training accuracy: {}-------->>auc:{}'.format(train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Pre-Training Test accuracy: {}------------>>auc:{}'.format(test_acc, test_auc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0
    global_weight_collector = list(global_net.cuda().parameters())


    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out,_ = net(x)
            loss = criterion(out, target)

            # for fedprox
            fed_prox_reg = 0.0
            # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)
    if write_log:
        logger.info('>> Training accuracy: {}---->>auc:{}'.format(train_acc, train_auc))
        logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
        logger.info('>> Test accuracy: {}----->>auc:{}'.format(test_acc, test_auc))
        logger.info(' ** Training complete **')
    print('>> Training accuracy: %f------->>auc:%f' % (train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Test accuracy: %f----------->>auc:%f' % (test_acc, test_auc))
    net.to('cpu')

    print(' ** Training complete **')
    return train_acc, val_acc, test_acc, train_auc, val_auc, test_auc

def train_net_HarmoFL(net_id, net, global_net, train_dataloader,val_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      device="cpu",write_log=True):
    '''
    use for HarmoFL
    '''
    from HarmoFL_utils.weight_perturbation import WPOptim
    # global_net.to(device)
    # net = nn.DataParallel(net)
    net.cuda()
    # else:
    if write_log:
        logger.info('Training network %s' % str(net_id))
        logger.info('n_training: %d' % len(train_dataloader))
        logger.info('n_test: %d' % len(test_dataloader))
    print('Training network %s' % str(net_id))
    print('n_training: %d' % len(train_dataloader))
    print('n_test: %d' % len(test_dataloader))
    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:  # 有些数据没有验证集
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    else:
        val_acc, val_auc = None, None
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)

    logger.info('>> Pre-Training Training accuracy: {}--->>auc:{}'.format(train_acc, train_auc))

    logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    logger.info('>> Pre-Training Test accuracy: {}------>>auc:{}'.format(test_acc, test_auc))
    print('>> Pre-Training Training accuracy: {}-------->>auc:{}'.format(train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Pre-Training Test accuracy: {}------------>>auc:{}'.format(test_acc, test_auc))

    if args.dataset == 'prostate':
        optimizer = WPOptim(params=net.parameters(), base_optimizer=optim.Adam, lr=args.lr, alpha=args.alpha, weight_decay=1e-4)
    else:
        optimizer = WPOptim(params=net.parameters(), base_optimizer=optim.SGD, lr=args.lr, alpha=args.alpha, momentum=0.9, weight_decay=1e-4)


    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0



    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out,_ = net(x)
            loss = criterion(out, target)


            loss.backward()
            optimizer.generate_delta(zero_grad=True)
            out, _ = net(x)
            criterion(out, target).backward()
            optimizer.step(zero_grad=True)

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)
    if write_log:
        logger.info('>> Training accuracy: {}---->>auc:{}'.format(train_acc, train_auc))
        logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
        logger.info('>> Test accuracy: {}----->>auc:{}'.format(test_acc, test_auc))
        logger.info(' ** Training complete **')
    print('>> Training accuracy: %f------->>auc:%f' % (train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Test accuracy: %f----------->>auc:%f' % (test_acc, test_auc))
    net.to('cpu')

    print(' ** Training complete **')
    return train_acc, val_acc, test_acc, train_auc, val_auc, test_auc
def train_net_fedcon(net_id, net, global_net, previous_nets, train_dataloader,val_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
                      round, device="cpu",write_log=True):
    # net = nn.DataParallel(net)
    net.cuda()
    if write_log:
        logger.info('Training network %s' % str(net_id))
        logger.info('n_training: %d' % len(train_dataloader))
        logger.info('n_test: %d' % len(test_dataloader))
    print('Training network %s' % str(net_id))
    print('n_training: %d' % len(train_dataloader))
    print('n_test: %d' % len(test_dataloader))
    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:  # 有些数据没有验证集
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    else:
        val_acc, val_auc = None, None
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)

    logger.info('>> Pre-Training Training accuracy: {}--->>auc:{}'.format(train_acc, train_auc))

    logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    logger.info('>> Pre-Training Test accuracy: {}------>>auc:{}'.format(test_acc, test_auc))
    print('>> Pre-Training Training accuracy: {}-------->>auc:{}'.format(train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Pre-Training Test accuracy: {}------------>>auc:{}'.format(test_acc, test_auc))


    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()
    # global_net.to(device)

    for previous_net in previous_nets:
        previous_net.cuda()
    global_w = global_net.state_dict()

    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out, pro1  = net(x)
            _, pro2 = global_net(x)

            posi = cos(pro1[-1], pro2[-1])
            logits = posi.reshape(-1,1)

            for previous_net in previous_nets:
                previous_net.cuda()
                _, pro3 = previous_net(x)
                nega = cos(pro1[-1], pro3[-1])
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                previous_net.to('cpu')

            logits /= temperature
            labels = torch.zeros(x.size(0)).cuda().long()

            loss2 = mu * criterion(logits, labels)


            loss1 = criterion(out, target)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))


    for previous_net in previous_nets:
        previous_net.to('cpu')
    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)
    if write_log:
        logger.info('>> Training accuracy: {}---->>auc:{}'.format(train_acc, train_auc))
        logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
        logger.info('>> Test accuracy: {}----->>auc:{}'.format(test_acc, test_auc))
        logger.info(' ** Training complete **')
    print('>> Training accuracy: %f------->>auc:%f' % (train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Test accuracy: %f----------->>auc:%f' % (test_acc, test_auc))
    net.to('cpu')

    print(' ** Training complete **')
    return train_acc, val_acc, test_acc, train_auc, val_auc, test_auc
def train_net_fedlwt(net_id, net,#target_model
                     source_optimizer,target_optimizer,wnet,lwnet,target_branch,
                     global_net, #source_model
                     previous_nets, train_dataloader,val_dataloader, test_dataloader, epochs,
                     lr, args_optimizer, mu, temperature, args,round, device="cpu",write_log=True):

    net.cuda()
    # local_models = args.sites.copy()
    server_model_name = [args.sites[args.global_center_idx]][0]
    # local_models.pop(args.sites[args.global_center_idx])
    if write_log:
        logger.info('Training network %s' % str(net_id))
        logger.info('n_training: %d' % len(train_dataloader))
        logger.info('n_test: %d' % len(test_dataloader))
    print('Training network %s' % str(net_id))
    print('n_training: %d' % len(train_dataloader))
    print('n_test: %d' % len(test_dataloader))
    train_acc, _ ,train_auc= compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:#有些数据没有验证集
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                          device=device)
    else:
        val_acc, val_auc=None,None
    test_acc, conf_matrix,_,test_auc= compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}--->>auc:{}'.format(train_acc,train_auc))

    logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    logger.info('>> Pre-Training Test accuracy: {}------>>auc:{}'.format(test_acc,test_auc))
    print('>> Pre-Training Training accuracy: {}-------->>auc:{}'.format(train_acc,train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Pre-Training Test accuracy: {}------------>>auc:{}'.format(test_acc,test_auc))
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.momentum,
                              weight_decay=args.reg)
    elif args_optimizer=="WPOtim":
        optimizer = WPOptim(params=net.parameters(), base_optimizer=optim.SGD, lr=lr, alpha=args.alpha,momentum=args.momentum, weight_decay=args.reg)

    for previous_net in previous_nets:
        previous_net.cuda()
    global_w = global_net.state_dict()
    # if args.loss_func == "crossentropy":
    #     criterion = nn.CrossEntropyLoss().cuda()
    # elif args.loss_func == "focalloss":
    criterion = FocalLoss().cuda()
    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1)
    state=dict()
    # mu = 0.001
    for epoch in range(epochs):
        state['epoch'] = epoch
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        epoch_transfer_loss_collector = []
        net.train()  # target_model
        global_net.to(device) #source_model
        global_net.eval()
        for batch_idx, data in enumerate(train_dataloader):
            ####迁移##########
            state['iter'] = batch_idx
            target_optimizer.zero_grad()

            loss = inner_objective(data,args, net, global_net, wnet,lwnet,
                                   target_branch,state=state,logger=logger,
                                   source_model_name=server_model_name,target_model_name=net_id, device=device)
            loss.backward()
            target_optimizer.step(None)

            for _ in range(args.T):
                target_optimizer.zero_grad()
                target_optimizer.step(inner_objective, data, args, net, global_net, wnet,lwnet,
                                   target_branch,state,logger,server_model_name,net_id, True)
            target_optimizer.zero_grad()
            target_optimizer.step(outer_objective, data,args, net, state)
            target_optimizer.zero_grad()
            source_optimizer.zero_grad()
            loss = outer_objective(data,args, net, state,device=device)
            loss.backward()
            target_optimizer.meta_backward()
            source_optimizer.step()
            epoch_transfer_loss_collector.append(loss)

            ####contrast_learning#######
            x, target = data[0].cuda(), data[1].cuda()
            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()
            out,pro1  = net(x)
            _, pro2= global_net(x)
            posi = cos(pro1[-1], pro2[-1])#求每个模型与全局模型的余弦相似度
            logits = posi.reshape(-1,1)
            for previous_net in previous_nets:
                previous_net.cuda()
                _, pro3 = previous_net(x)
                nega = cos(pro1[-1], pro3[-1])#计算当前模型与之前各个模型的余弦相似度
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)
                previous_net.to('cpu')
            logits /= temperature
            labels = torch.zeros(x.size(0)).cuda().long()
            loss2 = mu * criterion(logits, labels)  # 对比损失
            loss1 = criterion(out, target)  # 当前网络交叉熵损失
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())
        # print(epoch,sim_dict)
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        epoch_transfer_loss = sum(epoch_transfer_loss_collector) / len(epoch_transfer_loss_collector)
        if write_log:
            logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f transfer_loss:%f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2,epoch_transfer_loss))
        print('Epoch: %d Loss: %f Loss1: %f Loss2: %f transfer_loss:%f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2,epoch_transfer_loss))

    for previous_net in previous_nets:
        previous_net.to('cpu')
    train_acc, _ ,train_auc= compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                            device=device)
    test_acc, conf_matrix, _,test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    if write_log:
        logger.info('>> Training accuracy: {}---->>auc:{}'.format(train_acc,train_auc))
        logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
        logger.info('>> Test accuracy: {}----->>auc:{}' .format(test_acc,test_auc))
        logger.info(' ** Training complete **')
    print('>> Training accuracy: %f------->>auc:%f' % (train_acc,train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Test accuracy: %f----------->>auc:%f' % (test_acc,test_auc))
    net.to('cpu')

    print(' ** Training complete **')
    return train_acc,val_acc,test_acc,train_auc,val_auc,test_auc
def global_train_net(args,
                     source_optimizers_dict,target_optimizers_dict,
                     wnet_dict,lwnet_dict,target_branch_dict,
                     nets,
                     global_model=None,train_global_dl=None,val_global_dl=None,test_global_dl=None,device="cpu",write_log=True):
#训练全局模型，使用联邦学习算法（如FedLWT），同时将源模型迁移到目标模型。
    if args.alg=="fedlwt":
        server_model_name = [args.sites[args.global_center_idx]][0]
    else:
        server_model_name = "server"
    if global_model:
        global_model.cuda()
    if write_log:#打印全局模型的训练信息，如训练集大小、验证集大小和测试集大小
        logger.info('global_model_Training network 1' )
        logger.info('n_training: %d' % len(train_global_dl))
        logger.info('n_test: %d' % len(test_global_dl))
    state=dict()#用于存储当前训练状态的信息
    for epoch in range(args.epochs):
        state['epoch']=epoch
        # torch.cuda.empty_cache()
        for source_model_name,source_model in nets.items():
            if source_model_name==server_model_name:
                continue
            epoch_loss1_collector = []
            global_model.train()#target_model
            source_model.to(device)
            source_model.eval()
            dict_name="%sto%s"%(source_model_name,server_model_name)
            target_optimizer=target_optimizers_dict[dict_name]
            source_optimizer=source_optimizers_dict[dict_name]
            for batch_idx, data in enumerate(train_global_dl):
                state['iter']=batch_idx
                target_optimizer.zero_grad()
                loss=inner_objective(data,args,global_model,source_model,wnet_dict[dict_name],lwnet_dict[dict_name],target_branch_dict[dict_name],
                state,logger=logger,source_model_name=source_model_name,target_model_name=server_model_name,device=device)
                loss.backward()
                target_optimizer.step(None)

                for _ in range(args.T):
                    target_optimizer.zero_grad()
                    target_optimizer.step(inner_objective, data,args,global_model,source_model,
                                          wnet_dict[dict_name],lwnet_dict[dict_name],
                                     target_branch_dict[dict_name],state,logger,source_model_name,server_model_name, True)
                target_optimizer.zero_grad()
                target_optimizer.step(outer_objective, data,args,global_model,state)
                target_optimizer.zero_grad()
                source_optimizer.zero_grad()
                loss=outer_objective(data,args,global_model,state,device=device)
                loss.backward()
                target_optimizer.meta_backward()
                source_optimizer.step()
                epoch_loss1_collector.append(loss)
            epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
            if write_log:
                logger.info('%s model transfer to %s_Epoch: %d  Loss1: %f ' % (source_model_name,server_model_name,epoch,epoch_loss1))
            print('%s model transfer to %s_Epoch: %d  Loss1: %f ' % (source_model_name,server_model_name,epoch,epoch_loss1))
    train_acc, _, train_auc = compute_accuracy(global_model, train_global_dl, device=device)
    if val_global_dl is not None:
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(global_model, val_global_dl,
                                                              get_confusion_matrix=True,
                                                              device=device)
    else:
        val_acc,val_auc=None,None
    test_acc, conf_matrix, _, test_auc = compute_accuracy(global_model, test_global_dl,
                                                          get_confusion_matrix=True,
                                                          device=device)
    if write_log:
        logger.info('>> global_model_Training accuracy: {}---->>auc:{}'.format(train_acc, train_auc))

        logger.info('>> global_model_Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
        logger.info('>> global_model_Test accuracy: {}----->>auc:{}'.format(test_acc, test_auc))
        logger.info(' ** Training complete **')
    print('>> global_model_Training accuracy: %f------->>auc:%f' % (train_acc,train_auc))
    print('>> global_model_Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> global_model_Test accuracy: %f----------->>auc:%f' % (test_acc,test_auc))
    global_model.to('cpu')

    print(' ** Training complete **')
    return train_acc,val_acc,test_acc,train_auc,val_auc,test_auc
def local_train_net(nets, args,
                    source_optimizers_dict=None,target_optimizers_dict=None,
                    wnet_dict=None,lwnet_dict=None,target_branch_dict=None,
                    train_dl=None,val_dl=None, test_dl=None,
                    global_model = None, prev_model_pool = None, server_c = None, clients_c = None, round=None, device="cpu",write_log=True):
    if args.alg=="fedlwt":
        local_models = args.sites.copy()
        server_model_name = [args.sites[args.global_center_idx]][0]
        local_models.pop(args.global_center_idx)
    else:
        local_models = args.sites.copy()
        server_model_name = "server"

    avg_acc = 0.0
    acc_list = []
    auc={}
    if global_model:
        global_model.cuda()
    if server_c:
        server_c.cuda()
        server_c_collector = list(server_c.cuda().parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)
    for idx,(net_id, net) in enumerate(nets.items()):
        # dataidxs = net_dataidx_map[net_id]
        # dataidxs=int(net_id)
        if write_log:
            logger.info("Training network %s. batch_id: %s" % (str(net_id), str(net_id)))
        print("Training network %s. batch_id: %s" % (str(net_id),str(net_id) ))
        train_dl_local=train_dl[idx]
        if val_dl is not None:
            val_dl_local=val_dl[idx]
        else:
            val_dl_local=None
        test_dl_local=test_dl[idx]
        n_epoch = args.epochs

        if args.alg == 'fedlwt':
            #####迁移需要用的
            source_optimizer = source_optimizers_dict["%sto%s" % (server_model_name, net_id)]
            target_optimizer = target_optimizers_dict["%sto%s" % (server_model_name, net_id)]
            wnet = wnet_dict["%sto%s" % (server_model_name, net_id)]
            lwnet = lwnet_dict["%sto%s" % (server_model_name, net_id)]
            target_branch = target_branch_dict["%sto%s" % (server_model_name, net_id)]
            ##############
            prev_models=[]
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            train_acc, val_acc,test_acc,train_auc,val_auc,test_auc= train_net_fedlwt(net_id, net,
                                                                   source_optimizer, target_optimizer,
                                                                   wnet, lwnet, target_branch,
                                                                   global_model, prev_models,
                                                                   train_dl_local,val_dl_local, test_dl_local,
                                                    n_epoch,args.lr,args.optimizer, args.mu, args.temperature, args, round, device=device,write_log=write_log)
            auc[net_id]=[train_auc,test_auc]
        elif args.alg == 'fedavg':
            train_acc, val_acc, test_acc, train_auc, val_auc, test_auc = \
                train_net(net_id, net, train_dl_local,val_dl_local, test_dl_local, n_epoch, args.lr, args.optimizer,
                                          args,device=device)
            auc[net_id] = [train_auc, test_auc]
        elif args.alg == 'fedprox':
            train_acc, val_acc, test_acc, train_auc, val_auc, test_auc = \
                train_net_fedprox(net_id, net, global_model, train_dl_local,val_dl_local, test_dl_local, n_epoch,
                                                  args.lr,args.optimizer, args.mu, args, device=device)
            auc[net_id] = [train_auc, test_auc]
        elif args.alg == 'moon':
            prev_models = []
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            train_acc, val_acc, test_acc, train_auc, val_auc, test_auc = \
                train_net_fedcon(net_id, net, global_model, prev_models, train_dl_local,val_dl_local, test_dl_local,
                            n_epoch, args.lr,args.optimizer, args.mu, args.temperature, args, round,device=device)
            auc[net_id] = [train_auc, test_auc]
        elif args.alg=="HarmoFL":
            train_acc, val_acc, test_acc, train_auc, val_auc, test_auc = \
                train_net_HarmoFL(net_id, net, global_model, train_dl_local, val_dl_local, test_dl_local, n_epoch,
                                  args.lr, args.optimizer, args.mu, args, device=device)
        if write_log:
            logger.info("net %s final test acc %f" % (net_id, test_acc))
        print("net %s final test acc %f" % (net_id, test_acc))
        avg_acc += test_acc
        acc_list.append(test_acc)
    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        logger.info("std acc %f" % np.std(acc_list))
    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
    return nets,auc


if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)#使用mkdirs()函数在args.logdir和args.modeldir中创建目录用于存储日志文件和模型文件
    mkdirs(args.modeldir)
    if args.log_file_name is None:#如果未提供args.log_file_name，则使用当前时间戳生成一个文件名，否则使用指定的文件名
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:#移除日志记录器中的所有处理器，以便重新设置日志记录器
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')#设置日志记录器的配置，包括文件名、格式和级别

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    ###########################data loading

    sites, _, _, train_loaders, val_loaders, test_loaders,net_dataidx_map = \
        recorded_multicenters_data_API_dataloader(args)
    # print("len train_dl_global:", len(train_ds_global))
    ######分离全局模型数据和本地模型数据
    if args.alg == "fedlwt":  # n个本地模型，1个全局模型
        args.sites = sites
        train_dl_global = train_loaders[args.global_center_idx]
        train_loaders.pop(args.global_center_idx)
        if len(val_loaders) !=0:
            val_dl_global = val_loaders[args.global_center_idx]
            val_loaders.pop(args.global_center_idx)
        else:
            val_dl_global=None
            val_loaders=None
        test_dl_global = test_loaders[args.global_center_idx]
        test_loaders.pop(args.global_center_idx)

        n_party_per_round = int(args.n_parties * args.sample_fraction)
        party_list=sites.copy()
        party_list.pop(args.global_center_idx)
    else:#n+1个本地模型
        args.sites = sites
        if len(val_loaders)==0:
            val_dl_global=None
            val_loaders=None
        n_party_per_round = int(args.n_parties * args.sample_fraction)
        party_list = sites
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    train_dl=None
    data_size = len(test_loaders[0])
    test_dl=None
    logger.info("Initializing nets")
    print("Initializing nets")
    #=================net_init=========================
    if args.alg == "fedlwt":#n+1个本地模型，其他一个充当中心模型
        nets, local_model_meta_data, layer_type = init_nets(args, device='cpu')

        global_models, global_model_meta_data, global_layer_type = init_nets(args, device='cpu',server=True)
        global_model = global_models[args.sites[args.global_center_idx]]
    else:#n个本地模型，1个全局模型
        nets, local_model_meta_data, layer_type = init_nets(args, device='cpu')

        global_models, global_model_meta_data, global_layer_type = init_nets(args, device='cpu', server=True)
        global_model = global_models["server"]

    # nets[]
    n_comm_rounds = args.comm_round#联邦学习中的模型加载和参数更新
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:#创建一个moment_v，用于存储服务器的动量参数。然后遍历global_model的参数，将每个参数的值设置为0。这样可以实现服务器的动量更新。
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0

    if args.alg == 'fedlwt':
        # ==================meta net init===================
        pairs = init_pairs(args)
        ###全局模型向本地模型迁移用的meta_model 4个,本地模型向全局模型迁移 4个
        wnet_dict, lwnet_dict, weight_params_dict, target_params_dict, target_branch_dict = init_meta_model(args, pairs,
                                                                                                            global_models,
                                                                                                            nets)  # nets or global_model
        nets[args.sites[args.global_center_idx]] = global_model
        source_optimizers_dict, target_optimizers_dict = optimizer_init(args, weight_params_dict, nets,
                                                                        target_params_dict, target_branch_dict)
        nets.pop(args.sites[args.global_center_idx])
        old_nets_pool = []#用于存储预训练模型池
        if args.load_pool_file:
            for nets_id in range(args.model_buffer_size):
                old_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
                checkpoint = torch.load(args.load_pool_file)
                for net_id, net in old_nets.items():
                    net.load_state_dict(checkpoint['pool' + str(nets_id) + '_'+'net'+str(net_id)])
                old_nets_pool.append(old_nets)
        elif args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False

        for round in range(n_comm_rounds):
            time1=time.time()
            logger.info("in comm round:" + str(round))
            print("in comm round:" + str(round))#记录当前轮次的时间，并打印相关信息
            global_model.train()#将全局模型设置为训练状态，并将所有参数设置为可训练状态
            for param in global_model.parameters():
                param.requires_grad = True
#调用global_train_net函数，使用全局模型和相关的数据集（训练、验证和测试）进行训练，并获取训练、验证和测试的准确率以及AUC值
            global_train_acc,global_val_acc,global_test_acc,global_train_auc,global_val_auc,global_test_auc=\
                                                                global_train_net(args,
                                                                source_optimizers_dict,target_optimizers_dict,
                                                                wnet_dict,lwnet_dict,target_branch_dict,nets,
                                                                global_model=global_model,
                                                                train_global_dl=train_dl_global,val_global_dl=val_dl_global,test_global_dl=test_dl_global,device=device,write_log=True)
            # torch.cuda.empty_cache()
            party_list_this_round = party_list_rounds[round]#获取当前轮次需要参与通信的节点列表

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False#将全局模型设置为评估状态，并将所有参数设置为不可训练状态
            global_w = global_model.state_dict()#获取当前全局模型的参数（global_w），并记录一份当前全局模型的参数（old_global_w）
            old_global_w=global_w
            if args.server_momentum:#如果使用了动量，则还需要记录一份上一轮全局模型的参数（old_w）
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}#根据当前轮次需要参与通信的节点列表（party_list_this_round），获取这些节点对应的局部模型（nets_this_round）
            #注释掉直接替换参数的迁移方式
            # for net in nets_this_round.values():
            #     net.load_state_dict(global_w)

            # torch.cuda.empty_cache()

            _,local_auc=local_train_net(nets_this_round, args,
                                        source_optimizers_dict=source_optimizers_dict, target_optimizers_dict=target_optimizers_dict,
                                        wnet_dict=wnet_dict, lwnet_dict=lwnet_dict, target_branch_dict=target_branch_dict,
                                train_dl=train_loaders,val_dl=val_loaders, test_dl=test_loaders,
                            global_model = global_model, prev_model_pool=old_nets_pool, round=round, device=device,write_log=True)
            # torch.cuda.empty_cache()


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]#更新动量
                    global_w[key] = old_w[key] - moment_v[key]#更新后的动量应用于全局权重

            global_model.load_state_dict(global_w)#加载更新后的全局权重
            #summary(global_model.to(device), (3, 32, 32))

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))
            print('global n_training: %d' % len(train_dl_global))
            print('global n_test: %d' % len(test_dl_global))
            global_model.cuda()
            train_acc, train_loss,train_auc = compute_accuracy(global_model, train_dl_global, device=device)#计算训练集的准确性、损失和AUC
            if val_dl_global is not None:
                val_acc, conf_matrix_val, _, val_auc = compute_accuracy(global_model, val_dl_global,
                                                                        get_confusion_matrix=True,
                                                                        device=device)
            else:
                val_acc,val_auc=None,None
            test_acc, conf_matrix, _ ,test_auc= compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)
            global_model.to('cpu')
            logger.info('>> Global Model Train accuracy: {}----->>auc:{}' .format(train_acc,train_auc))
            logger.info('>> global_model_Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
            logger.info('>> Global Model Test accuracy: {}------->>auc:{}' .format(test_acc,test_auc))
            logger.info('>> Global Model Train loss: %f' % train_loss)
            print('>> Global Model Train accuracy: %f' % train_acc)
            print('>> Global Model val accuracy: {}' .format(val_acc))
            print('>> Global Model Test accuracy: %f' % test_acc)
            print('>> Global Model Train loss: %f' % train_loss)
            if len(old_nets_pool) < args.model_buffer_size:#检查当前旧网络池的大小是否小于预定义的模型缓冲区大小
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()#对每个网络，将其转换为评估模式
                    for param in net.parameters():#将所有参数的requires_grad设置为False，以防止在训练过程中更新它们
                        param.requires_grad = False
                old_nets_pool.append(old_nets)#将复制的旧网络添加到旧网络池中
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size-2, -1, -1):#倒序遍历旧网络池,并将每个旧网络复制到下一个索引
                    old_nets_pool[i] = old_nets_pool[i+1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets#将新复制的旧网络添加到旧网络池的最后一个索引
            if round%each_save_interval==0:#查当前轮数（round）是否可以被each_save_interval整除
                if args.save_model:
                    mkdirs(args.modeldir + '%s/'%args.alg)
                    torch.save(global_model.state_dict(), args.modeldir+'%s/global_model_%d'%(args.alg,round)+args.log_file_name+'.pth')#保存全局模型
                    for k,net_id in enumerate(nets):#遍历nets字典，保存每个本地模型
                        torch.save(nets[net_id].state_dict(), args.modeldir+'%s/localmodel_%s_%d'%(args.alg,net_id,round)+args.log_file_name+'.pth')
                    for k,net_id in enumerate(wnet_dict):#遍历wnet_dict字典，保存每个边缘模型
                        torch.save(wnet_dict[net_id].state_dict(), args.modeldir + '%s/wnet_model_%s_%d' % (args.alg,
                        net_id, round) + args.log_file_name + '.pth')
                    for k,net_id in enumerate(lwnet_dict):#遍历lwnet_dict字典，保存每个局部边缘模型
                        torch.save(lwnet_dict[net_id].state_dict(), args.modeldir + '%s/lwnet_model_%s_%d' % (args.alg,
                        net_id, round) + args.log_file_name + '.pth')
                    for nets_id, old_nets in enumerate(old_nets_pool):
                        torch.save({'pool'+ str(nets_id) + '_'+'net'+str(net_id): net.state_dict() for net_id, net in old_nets.items()}, args.modeldir+'%s/prev_model_pool_%d'%(args.alg,round)+args.log_file_name+'.pth')
            time2=time.time()
            print("round:%d,consume:%s minutes"%(round,(time2-time1)/60.0))

    elif args.alg == 'moon':
        old_nets_pool = []
        if args.load_pool_file:
            for nets_id in range(args.model_buffer_size):
                old_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
                checkpoint = torch.load(args.load_pool_file)
                for net_id, net in old_nets.items():
                    net.load_state_dict(checkpoint['pool' + str(nets_id) + '_'+'net'+str(net_id)])
                old_nets_pool.append(old_nets)
        elif args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, train_dl=train_loaders,val_dl=val_loaders, test_dl=test_loaders, global_model = global_model, prev_model_pool=old_nets_pool, round=round, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]


            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            #summary(global_model.to(device), (3, 32, 32))

            # logger.info('global n_training: %d' % len(train_dl_global))
            # logger.info('global n_test: %d' % len(test_dl))
            # global_model.cuda()
            # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            # test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            # global_model.to('cpu')
            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            # logger.info('>> Global Model Test accuracy: %f' % test_acc)
            # logger.info('>> Global Model Train loss: %f' % train_loss)

            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size-2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i+1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets


            if round%each_save_interval==0:
                if args.save_model:
                    mkdirs(args.modeldir + '%s/'%args.alg)
                    torch.save(global_model.state_dict(), args.modeldir+'%s/global_model_%d'%(args.alg,round)+args.log_file_name+'.pth')
                    for k,net_id in enumerate(nets):
                        torch.save(nets[net_id].state_dict(), args.modeldir+'%s/localmodel_%s_%d'%(args.alg,net_id,round)+args.log_file_name+'.pth')
                    for nets_id, old_nets in enumerate(old_nets_pool):
                        torch.save({'pool'+ str(nets_id) + '_'+'net'+str(net_id): net.state_dict() for net_id, net in old_nets.items()}, args.modeldir+'%s/prev_model_pool_%d'%(args.alg,round)+args.log_file_name+'.pth')


    elif args.alg == 'fedavg':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, train_dl=train_loaders,val_dl=val_loaders, test_dl=test_loaders, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)

            #logger.info('global n_training: %d' % len(train_dl_global))
            # logger.info('global n_test: %d' % len(test_dl))
            # global_model.cuda()
            # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            # test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            #
            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            # logger.info('>> Global Model Test accuracy: %f' % test_acc)
            # logger.info('>> Global Model Train loss: %f' % train_loss)

            if round%each_save_interval==0:
                if args.save_model:
                    mkdirs(args.modeldir + '%s/'%args.alg)
                    torch.save(global_model.state_dict(), args.modeldir+'%s/global_model_%d'%(args.alg,round)+args.log_file_name+'.pth')
                    for k,net_id in enumerate(nets):
                        torch.save(nets[net_id].state_dict(), args.modeldir+'%s/localmodel_%s_%d'%(args.alg,net_id,round)+args.log_file_name+'.pth')

    elif args.alg == 'fedprox':

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, train_dl=train_loaders,val_dl=val_loaders, test_dl=test_loaders,global_model = global_model, device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            global_model.load_state_dict(global_w)



            mkdirs(args.modeldir + 'fedprox/')
            # global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir +'fedprox/'+args.log_file_name+ '.pth')
            if round%each_save_interval==0:
                if args.save_model:
                    mkdirs(args.modeldir + '%s/'%args.alg)
                    torch.save(global_model.state_dict(), args.modeldir+'%s/global_model_%d'%(args.alg,round)+args.log_file_name+'.pth')
                    for k,net_id in enumerate(nets):
                        torch.save(nets[net_id].state_dict(), args.modeldir+'%s/localmodel_%s_%d'%(args.alg,net_id,round)+args.log_file_name+'.pth')
    elif args.alg == 'HarmoFL':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, train_dl=train_loaders, val_dl=val_loaders, test_dl=test_loaders,
                            global_model=global_model, device=device)
            global_model.to('cpu')

            # update global model
            # total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            # fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
            client_num = len(party_list_this_round)
            fed_avg_freqs = [1. / client_num for i in range(client_num)]
            # print(fed_avg_freqs)
            for key in global_model.state_dict().keys():
                if "num_batches_tracked" in key:
                    continue
                temp = torch.zeros_like(global_model.state_dict()[key])
                for client_idx in range(len(fed_avg_freqs)):
                    temp += fed_avg_freqs[client_idx] * nets_this_round[client_idx].state_dict()[key]
                global_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(len(fed_avg_freqs)):
                    nets_this_round[client_idx].state_dict()[key].data.copy_(global_model.state_dict()[key])
                if 'running_amp' in key:
                    # aggregate at first round only to save communication cost
                    global_model.amp_norm.fix_amp = True
                    for model in nets_this_round:
                        model.amp_norm.fix_amp = True

            mkdirs(args.modeldir + 'HarmoFL/')
            # global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir + 'HarmoFL/' + args.log_file_name + '.pth')
            if round % each_save_interval == 0:
                if args.save_model:
                    mkdirs(args.modeldir + '%s/' % args.alg)
                    torch.save(global_model.state_dict(),
                               args.modeldir + '%s/global_model_%d' % (args.alg, round) + args.log_file_name + '.pth')
                    for k, net_id in enumerate(nets):
                        torch.save(nets[net_id].state_dict(), args.modeldir + '%s/localmodel_%s_%d' % (
                        args.alg, net_id, round) + args.log_file_name + '.pth')


