# xujun 2023.10.18
import argparse
import os

model="resnet18"
model1="vgg16"

dataset="早期肺癌"#改为对应数据集名"早期肺癌"
global_center_idx=0#选择一个中心数据作为服务器模型的数据，这里选择第一个数据样本作为服务器模型的数据。
batch_size=32#批次大小
epochs=1#本地模型循环，表示在每个客户端模型更新之前，需要训练本地模型的次数
fed_structure="fedlwt" #n个party/n+1个 fed 表示联邦学习，lwt 表示局部权重传输
comm_round=10#框架大循环，表示联邦学习框架中的通信轮数
init_seed=4000#随机数种子
input_shape=224 #32

current_dir = os.path.dirname(os.path.abspath(__file__))#获取main.py所在的文件夹路径
DATA_DIR=os.path.join(current_dir,r"G:\研究生\早期肺癌") #输入数据路径

save_roof = os.path.join(current_dir, r"G:\研究生\徐俊-异构联邦\1", fed_structure, model, dataset)
loss_function="focalloss"#"crossentropy or focalloss"在训练模型时，可以使用交叉熵损失函数（通常用于多分类问题）或焦点损失函数（通常用于目标检测问题）

if fed_structure=="fedavg":#联邦学习结构，这里设置为联邦平均
    server_momentum=1
else:
    server_momentum=0
# publish_data1=["camelyon17","prostate","nuclei"]
# publish_data2=["cifar100","cifar10","tiny-imagenet"]#lidc还没封装
# private_data=["lung_nodules","gastric","lidc"]
if fed_structure=="HarmoFL":#联邦学习结构，这里设置为混合联邦学习
    open_perturbe=True#是否开启干扰
else:
    open_perturbe = False

num_classes=2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=model, help='neural network used in training')
    parser.add_argument('--dataset', type=str, default=dataset, help='dataset_train used for training')
    parser.add_argument('--global_center_idx', type=int, default=global_center_idx, help='choose one center as the global')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))#神经网络配置，是一个列表，指定了每个隐藏层的节点数。通过将输入字符串使用逗号和空格分隔，并将其转换为整数列表的方式来传递该参数。
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy only use for cifar100,cifar10,tinyimagenet,lidc:homo,iid,noniid-labeldir,noniid')#数据分割策略，用于指定如何划分数据集
    parser.add_argument('--batch-size', type=int, default=batch_size, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=epochs, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=4, help='number of workers in a distributed cluster')#分布式集群中的工作节点数量
    parser.add_argument('--alg', type=str, default=fed_structure,
                        help='communication strategy: fedavg/fedprox/moon/HarmoFL/fedlwt')#通信策略，用于指定联邦学习算法的选择
    parser.add_argument('--comm_round', type=int, default=comm_round, help='number of maximum communication roun')#最大通信轮数
    parser.add_argument('--init_seed', type=int, default=init_seed, help="Random seed")
    parser.add_argument('--input-shape', type=int, default=input_shape, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")#Dropout 概率,是一种正则化技术，用于减少神经网络的过拟合。
    parser.add_argument('--datadir', type=str, required=False, default=DATA_DIR, help="Data directory")
    parser.add_argument('--reg', type=float, default=0.0005, help="L2 regularization strength")#L2 正则化强度
    parser.add_argument('--logdir', type=str, required=False, default="%s/logs_auc_ratio/"%save_roof, help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="%s/models_auc_ratio/"%save_roof, help='Model directory path')
    parser.add_argument('--beta_distribution', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')#用于数据分区的狄利克雷分布的参数
    parser.add_argument('--device', type=str, default='cuda', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')#对于 fedprox 或 moon 算法的参数 mu
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')#投影层的输出维度
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')#对比损失函数的温度参数
    parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')#本地优化训练的最大轮数
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')#存储用于对比损失函数的前几个模型的数量
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')#模型池的选项
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')#每轮采样的客户端比例
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')#要加载的全局模型的文件名
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')#要加载的旧模型池的路径
    parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')#加载的模型已经执行的轮数
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')#是否加载第一个网络作为旧网络
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')#使用普通模型还是聚合模型
    parser.add_argument('--loss_func', type=str, default=loss_function,help="crossentropy or focalloss")#损失函数的类型
    parser.add_argument('--save_model',type=int,default=1)#是否保存模型,1保存
    parser.add_argument('--use_project_head', type=int, default=0)#是否使用投影头，1使用
    parser.add_argument('--server_momentum', type=float, default=server_momentum, help='the server momentum (FedAvgM)')#服务器动量（FedAvgM）参数
    parser.add_argument('--global_pre_join_with_new_global',type=float,default=3,help='devise the global_parameter_update')#全局参数更新策略的参数，用于设定全局参数在与新的全局模型聚合时的更新策略
    #######扰动参数#################
    parser.add_argument('--alpha', type=float, default=0.05, help='The hyper parameter of perturbation in HarmoFL')#扰动参数，在 HarmoFL（一种联邦学习算法）中用于调整模型之间的扰动程度。
    parser.add_argument('--open_perturbe', type=int, default=open_perturbe)#扰动是否开启
    ##########全局模型参数更新策略############
    parser.add_argument('--strategy', type=int, default=1, help='model_pre_matching')#全局模型参数更新策略，用于指定模型预匹配的策略。
    ##############异构参数################
    parser.add_argument('--num-classes', type=int, default=num_classes, help='the number of category')
    parser.add_argument('--source-model', default=model, type=str)# 源模型
    parser.add_argument('--source-domain', default='imagenet', type=str)#源域
    parser.add_argument('--source-path', type=str, default=None)
    parser.add_argument('--target-model', default=model1, type=str)#目标模型
    parser.add_argument('--weight-path', type=str, default=None)
    parser.add_argument('--wnet-path', type=str, default=None)#WNet 路径
    parser.add_argument('--open_lw2w', type=int, default=False, help='model_pre_matching')#是否启用 LW2W
    parser.add_argument('--nesterov', action='store_true')#是否使用 Nesterov 动量，用于指定是否在优化过程中使用 Nesterov 动量。
    parser.add_argument('--schedule', action='store_true', default=True)#是否使用学习率调度
    parser.add_argument('--beta', type=float, default=0.5)#用于调整一些算法中的超参数
    parser.add_argument('--pairs', type=str, default='4-4,4-3,4-2,4-1,3-4,3-3,3-2,3-1,2-4,2-3,2-2,2-1,1-4,1-3,1-2,1-1')#对应关系
    parser.add_argument('--meta-lr', type=float, default=1e-4, help='Initial learning rate for meta networks')#元网络的初始学习率
    parser.add_argument('--meta-wd', type=float, default=1e-4)#元网络的权重衰减
    parser.add_argument('--loss-weight', action='store_true', default=True)#是否使用损失权重
    parser.add_argument('--loss-weight-type', type=str, default='relu6')#损失权重类型
    parser.add_argument('--loss-weight-init', type=float, default=1.0)#损失权重的初始值
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--source-optimizer', type=str, default='sgd')#源优化器
    parser.add_argument('--experiment', default='logs', help='Where to store models')#实验目录，用于指定保存模型和实验结果的目录。
    parser.add_argument('--target-mhsa', type=bool, default=False,help="utilize the mhsa")#是否使用 Mhsa（Multi-head self-attention）目标模型
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')#动量参数，用于指定优化器的动量参数。
    parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay')#权重衰减
    ###harmFL
    parser.add_argument('--imbalance', action='store_true', help='do not truncate train data to same length')#是否对训练数据进行截断以保持相同长度，用于控制训练数据集的不平衡性问题，以保持相同的样本数或长度
    ###########################
    args = parser.parse_args()
    return args