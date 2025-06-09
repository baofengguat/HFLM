# xujun 2023.10.18
import argparse
import os

model="resnet18"
model1="vgg16"

dataset="1"
global_center_idx=0
batch_size=32
epochs=1
fed_structure="fedlwt"
comm_round=10
init_seed=4000
input_shape=224

current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(current_dir,r"G:")

save_roof = os.path.join(current_dir, r"G:", fed_structure, model, dataset)
loss_function="focalloss"

if fed_structure=="fedavg":
    server_momentum=1
else:
    server_momentum=0

num_classes=2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=model, help='neural network used in training')
    parser.add_argument('--dataset', type=str, default=dataset, help='dataset_train used for training')
    parser.add_argument('--global_center_idx', type=int, default=global_center_idx, help='choose one center as the global')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy only use for cifar100,cifar10,tinyimagenet,lidc:homo,iid,noniid-labeldir,noniid')
    parser.add_argument('--batch-size', type=int, default=batch_size, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=epochs, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=4, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default=fed_structure,
                        help='communication strategy: fedavg/fedprox/moon/HarmoFL/fedlwt')
    parser.add_argument('--comm_round', type=int, default=comm_round, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=init_seed, help="Random seed")
    parser.add_argument('--input-shape', type=int, default=input_shape, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default=DATA_DIR, help="Data directory")
    parser.add_argument('--reg', type=float, default=0.0005, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="%s/logs_auc_ratio/"%save_roof, help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="%s/models_auc_ratio/"%save_roof, help='Model directory path')
    parser.add_argument('--beta_distribution', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss_func', type=str, default=loss_function,help="crossentropy or focalloss")
    parser.add_argument('--save_model',type=int,default=1)
    parser.add_argument('--use_project_head', type=int, default=0)
    parser.add_argument('--server_momentum', type=float, default=server_momentum, help='the server momentum (FedAvgM)')
    parser.add_argument('--global_pre_join_with_new_global',type=float,default=3,help='devise the global_parameter_update')
    parser.add_argument('--alpha', type=float, default=0.05, help='The hyper parameter of perturbation in HarmoFL')
    parser.add_argument('--open_perturbe', type=int, default=open_perturbe)
    parser.add_argument('--strategy', type=int, default=1, help='model_pre_matching')
    parser.add_argument('--num-classes', type=int, default=num_classes, help='the number of category')
    parser.add_argument('--source-model', default=model, type=str)
    parser.add_argument('--source-domain', default='imagenet', type=str)
    parser.add_argument('--source-path', type=str, default=None)
    parser.add_argument('--target-model', default=model1, type=str)
    parser.add_argument('--weight-path', type=str, default=None)
    parser.add_argument('--wnet-path', type=str, default=None)
    parser.add_argument('--open_lw2w', type=int, default=False, help='model_pre_matching')
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--schedule', action='store_true', default=True)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--pairs', type=str, default='4-4,4-3,4-2,4-1,3-4,3-3,3-2,3-1,2-4,2-3,2-2,2-1,1-4,1-3,1-2,1-1')
    parser.add_argument('--meta-lr', type=float, default=1e-4, help='Initial learning rate for meta networks')
    parser.add_argument('--meta-wd', type=float, default=1e-4)
    parser.add_argument('--loss-weight', action='store_true', default=True)
    parser.add_argument('--loss-weight-type', type=str, default='relu6')
    parser.add_argument('--loss-weight-init', type=float, default=1.0)
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--source-optimizer', type=str, default='sgd')
    parser.add_argument('--experiment', default='logs', help='Where to store models')
    parser.add_argument('--target-mhsa', type=bool, default=False,help="utilize the mhsa")
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--imbalance', action='store_true', help='do not truncate train data to same length')
    args = parser.parse_args()
    return args