'''
use for federated learning multi-center data loading
the data includes:
camelyon17/prostate/Nuclei/5_hospital_lung_nodules/4_gastric_centers/cifar-100/cifar-10/tinyimage-net/LIDC
the API of lidc:unfinished
'''
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import SimpleITK as sitk
import random
import cv2
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from torchvision.datasets import ImageFolder, DatasetFolder, CIFAR10, CIFAR100
import math
import logging
import shutil
from torch.autograd import Variable
import torch.nn.functional as F
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)


####API
class Camelyon17(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=None):
        assert split in ['train', 'test']
        assert int(site) in [1, 2, 3, 4, 5]  # five hospital

        base_path = base_path if base_path is not None else 'G:\研究生\徐俊-异构联邦\模型'
        self.base_path = base_path

        data_dict = np.load(os.path.join(base_path,'data.pkl'), allow_pickle=True)
        self.paths, self.labels = data_dict[f'hospital{site}'][f'{split}']

        self.transform = transform
        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize((224, 224))
        if self.transform is not None:
            image = self.transform(image)

        return image, label

class Prostate(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=None):
        channels = {'BIDMC': 3, 'HK': 3, 'I2CVB': 3, 'ISBI': 3, 'ISBI_1.5': 3, 'UCL': 3}
        assert site in list(channels.keys())
        self.split = split

        base_path = base_path if base_path is not None else '../data/prostate'

        images, labels = [], []
        sitedir = os.path.join(base_path, site)

        ossitedir = np.load(os.path.join(base_path,"{}-dir.npy".format(site))).tolist()

        for sample in ossitedir:
            sampledir = os.path.join(sitedir, sample)
            if os.path.getsize(sampledir) < 1024 * 1024 and sampledir.endswith("nii.gz"):
                imgdir = os.path.join(sitedir, sample[:6] + ".nii.gz")
                label_v = sitk.ReadImage(sampledir)
                image_v = sitk.ReadImage(imgdir)
                label_v = sitk.GetArrayFromImage(label_v)
                label_v[label_v > 1] = 1
                image_v = sitk.GetArrayFromImage(image_v)
                image_v = convert_from_nii_to_png(image_v)

                for i in range(1, label_v.shape[0] - 1):
                    label = np.array(label_v[i, :, :])
                    if (np.all(label == 0)):
                        continue
                    image = np.array(image_v[i - 1:i + 2, :, :])
                    image = np.transpose(image, (1, 2, 0))

                    labels.append(label)
                    images.append(image)
        labels = np.array(labels).astype(int)
        images = np.array(images)

        index = np.load("../data/prostate/{}-index.npy".format(site)).tolist()

        labels = labels[index]
        images = images[index]

        trainlen = 0.8 * len(labels) * 0.8
        vallen = 0.8 * len(labels) - trainlen
        testlen = 0.2 * len(labels)

        if (split == 'train'):
            self.images, self.labels = images[:int(trainlen)], labels[:int(trainlen)]

        elif (split == 'val'):
            self.images, self.labels = images[int(trainlen):int(trainlen + vallen)], labels[int(trainlen):int(
                trainlen + vallen)]
        else:
            self.images, self.labels = images[int(trainlen + vallen):], labels[int(trainlen + vallen):]

        self.transform = transform
        self.channels = channels[site]
        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            if self.split == 'train':
                R1 = RandomRotate90()
                image, label = R1(image, label)
                R2 = RandomFlip()
                image, label = R2(image, label)

            image = np.transpose(image, (2, 0, 1))
            image = torch.Tensor(image)

            label = self.transform(label)

        return image, label

class Nuclei(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=None):
        assert split in ['train', 'val', 'test']
        assert site in ['SiteA', 'SiteB', 'SiteC', 'SiteD', 'SiteE', 'SiteF']

        self.base_path = base_path if base_path is not None else r'E:\\lusl\\external_data\\NucSeg'
        self.base_path = os.path.join(self.base_path, site)

        images = []
        labels = []

        if split == 'train' or split == 'val':
            self.base_path = os.path.join(self.base_path, split)
            img_path = os.path.join(self.base_path, "images")
            lbl_path = os.path.join(self.base_path, "labels")
            for i in os.listdir(img_path):
                img_dir = os.path.join(img_path, i)
                ibl_dir = os.path.join(lbl_path, i.split('.')[0] + ".png")
                images.append(img_dir)
                labels.append(ibl_dir)

            self.images, self.labels = images, labels

        elif split == 'test':
            self.base_path = os.path.join(self.base_path, "test")
            img_path = os.path.join(self.base_path, "images")
            lbl_path = os.path.join(self.base_path, "labels")
            for i in os.listdir(img_path):
                img_dir = os.path.join(img_path, i)
                ibl_dir = os.path.join(lbl_path, i.split('.')[0] + ".png")
                images.append(img_dir)
                labels.append(ibl_dir)

            self.images, self.labels = images, labels

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = cv2.imread(self.labels[idx], 0)
        image = Image.open(self.images[idx].replace("\\", "/")).convert('RGB')

        label[label == 255] = 1

        label = Image.fromarray(label)

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

            TTensor = transforms.ToTensor()
            image = TTensor(image)

            label = np.array(label)
            label = torch.Tensor(label)

            label = torch.unsqueeze(label, dim=0)

        return image, label


def get_lung_leison_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, center_name="江门医院"):
    if dataset == "lung_nodules":
        dl_obj = ImageFolder_custom
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_ds = dl_obj(datadir + '/%s/train_data/' % center_name, dataidxs=dataidxs,
                          transform=transform_train)  # server训练数据默认为江门医院
        test_ds = dl_obj(datadir + '/%s/test_data/' % center_name, transform=transform_test)
        print(f'[Client {center_name}] Train={len(train_ds)}, Test={len(test_ds)}')
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=8,
                                   pin_memory=True, prefetch_factor=4)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=8, pin_memory=True,
                                  prefetch_factor=4)
    else:
        raise
    return train_dl, test_dl, train_ds, test_ds

###CIFAR API
class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        # img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class CIFAR100_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir+'./train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir+'./val/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)

#####functional function
def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

class RandomRotate90:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            if mask is not None:
                mask = np.rot90(mask, factor)
        return img.copy(), mask.copy()


class RandomFlip:
    def __init__(self, prob=0.75):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)

        return img, mask

def convert_from_nii_to_png(img):
    high = np.quantile(img, 0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg = (newimg * 255).astype(np.uint8)
    return newimg


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0,center_name=None):
    if dataset in ('cifar10', 'cifar100'):
        if dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=noise_level),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated

            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(32),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     normalize
            # ])
            transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])



        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    elif dataset == 'tinyimagenet':
        dl_obj = ImageFolder_custom
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_ds = dl_obj(datadir+'./train/', dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(datadir+'./val/', transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    # elif dataset=="lung_nodules" or dataset=="lung_nodules_amp":
    #     dl_obj = ImageFolder_custom
    #     transform_train = transforms.Compose([
    #         transforms.ToTensor(),
    #         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         # AmpNorm((3,224,224))
    #     ])
    #     transform_test = transforms.Compose([
    #         transforms.ToTensor(),
    #         # AmpNorm((3,224,224))
    #         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ])
    #
    #     train_ds = dl_obj(datadir + '/%s/train_data/'%center_name, dataidxs=dataidxs, transform=transform_train)#server训练数据默认为江门医院
    #     test_ds = dl_obj(datadir + '/%s/test_data/'%center_name, transform=transform_test)
    #
    #     train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True,num_workers=8,pin_memory=True)
    #     test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False,num_workers=8,pin_memory=True)
    #
    # elif dataset == "gastric":
    #     dl_obj = ImageFolder_custom
    #     transform_train = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Resize([224, 224])
    #         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         # AmpNorm((3,224,224))
    #     ])
    #     transform_test = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Resize([224, 224])
    #         # AmpNorm((3,224,224))
    #         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ])
    #
    #     train_ds = dl_obj(datadir + '/%s/train_data/' % center_name, dataidxs=dataidxs,
    #                       transform=transform_train)  # server训练数据默认为江门医院
    #     test_ds = dl_obj(datadir + '/%s/test_data/' % center_name, transform=transform_test)
    #
    #     train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=8,
    #                                pin_memory=True)
    #     test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=8, pin_memory=True)
    # elif dataset == "lidc":
    #     dl_obj = ImageFolder_custom
    #     transform_train = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Resize([224, 224])
    #         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         # AmpNorm((3,224,224))
    #     ])
    #     transform_test = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Resize([224, 224])
    #         # AmpNorm((3,224,224))
    #         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ])
    #
    #     train_ds = dl_obj(datadir + '/%s/train_data/' % center_name, dataidxs=dataidxs,
    #                       transform=transform_train)  # server训练数据默认为江门医院
    #     test_ds = dl_obj(datadir + '/%s/test_data/' % center_name, transform=transform_test)
    #
    #     train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=8,
    #                                pin_memory=True)
    #     test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=8, pin_memory=True)
    return train_dl, test_dl, train_ds, test_ds


class med_DataSet(data.Dataset):
    def __init__(self, root=None,
                 data_set="",center="",transform=None):
        self.root = root
        self.files = []
        self.patient_img_num=[]
        self.transform=transform
        img_num=0
        leison1_list=["肺腺癌","1","复发","进展"]#针对肺结节/胃复发/LIDC整理数据/早期肺癌
        for leison_class in os.listdir(os.path.join(self.root,center,data_set)):
            for patient in os.listdir(os.path.join(self.root,center,data_set,leison_class)):
            # for split in ["train", "trainval", "val"]:
                for img_name in os.listdir(os.path.join(self.root,center,data_set,leison_class,patient)):
                    img_path=os.path.join(
                                self.root, center, data_set, leison_class, patient,img_name)
                    self.files.append({
                        "img_path": img_path,
                        "label": 1 if leison_class in leison1_list else 0,
                        "center": center,
                        "leison_class": leison_class,
                        "patient_name": patient})
    def __len__(self):
        return len(self.files)#定义__len__方法，返回数据集中的文件数量

    def __getitem__(self, index, img_path=None):
        datafiles = self.files[index]
        # patient_name = datafiles["patient_name"]
        # center_name = datafiles["center"]
        # leisonclass = datafiles["leison_class"]
        label = datafiles["label"]

        image = Image.open(datafiles["img_path"]).convert('RGB')

        image = image.resize((224, 224))
        if self.transform is not None:
            image = self.transform(image)
        return image,label




def recorded_multicenters_data_API_dataloader(args):
    '''the dataloader support for
    camelyon17/prostate/Nuclei/5_hospital_lung_nodules/4_gastric_centers/cifar-100/cifar-10/tinyimage-net/LIDC
    '''
    publish_data1=["camelyon17","prostate","nuclei"]
    publish_data2=["cifar100","cifar10","tiny-imagenet"]#lidc还没封装
    publish_data3=["lidc"]
    private_data=["lung_nodules","gastric","早期肺癌"] #对于新的数据，需要把名字加上，然后下面对应if处添加对应出来的程序
    train_loaders, test_loaders = [], []#目的是创建训练、验证和测试数据加载器
    val_loaders = []
    trainsets, testsets = [], []
    valsets = []
    sites=[]
    if args.dataset in publish_data1:
        if args.dataset == 'camelyon17':
            # args.lr = 1e-3
            # loss_fun = nn.CrossEntropyLoss()
            sites = ['1', '2', '3', '4', '5']
            net_dataidx_map = {}
            for site in sites:
                trainset = Camelyon17(site=site, split='train',base_path=args.datadir, transform=transforms.ToTensor())
                testset = Camelyon17(site=site, split='test',base_path=args.datadir, transform=transforms.ToTensor())
                val_len = math.floor(len(trainset) * 0.2)
                train_idx = list(range(len(trainset)))[:-val_len]
                val_idx = list(range(len(trainset)))[-val_len:]
                valset = torch.utils.data.Subset(trainset, val_idx)
                trainset = torch.utils.data.Subset(trainset, train_idx)
                print(f'[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
                trainsets.append(trainset)
                valsets.append(valset)
                testsets.append(testset)
                net_dataidx_map[site] = trainset
                # print(len(trainset))
        elif args.dataset == 'prostate':
            # args.lr = 1e-4
            # args.iters = 500
            # model = UNet(input_shape=[3, 384, 384])
            # loss_fun = JointLoss()
            sites = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']
            net_dataidx_map = {}
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            for site in sites:
                trainset = Prostate(site=site, split='train',base_path=args.datadir, transform=transform)
                valset = Prostate(site=site, split='val',base_path=args.datadir, transform=transform)
                testset = Prostate(site=site, split='test',base_path=args.datadir, transform=transform)

                print(f'[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
                trainsets.append(trainset)
                valsets.append(valset)
                testsets.append(testset)
                net_dataidx_map[site] = trainset
                print(len(trainset))
        elif args.dataset == 'nuclei':
            # args.lr = 1e-4
            # args.iters = 500
            args.imbalance = True
            # model = UNet(input_shape=[3, 256, 256])
            # loss_fun = DiceLoss()
            sites = ['SiteA', 'SiteB', 'SiteC', 'SiteD', 'SiteE', 'SiteF']
            net_dataidx_map = {}
            transform = transforms.Compose([
                transforms.Resize([256, 256]),
            ])
            for site in sites:
                trainset = Nuclei(site=site, split='train',base_path=args.datadir, transform=transform)
                valset = Nuclei(site=site, split='val',base_path=args.datadir, transform=transform)
                testset = Nuclei(site=site, split='test',base_path=args.datadir, transform=transform)
                print(f'[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
                trainsets.append(trainset)
                valsets.append(valset)
                testsets.append(testset)
                net_dataidx_map[site] = trainset
                print(len(trainset))

        min_data_len = min([len(s) for s in trainsets])
        for idx in range(len(trainsets)):
            if args.imbalance:
                trainset = trainsets[idx]
                valset = valsets[idx]
                testset = testsets[idx]
            else:
                trainset = torch.utils.data.Subset(trainsets[idx], list(range(int(min_data_len))))
                valset = valsets[idx]
                testset = testsets[idx]

            train_loaders.append(torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True))
            val_loaders.append(torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False))
            test_loaders.append(torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False))
        return sites, trainsets, testsets, train_loaders, val_loaders, test_loaders,net_dataidx_map
    elif args.dataset in publish_data2:
        if args.dataset == 'cifar10':
            X_train, y_train, X_test, y_test = load_cifar10_data(args.datadir)
        elif args.dataset == 'cifar100':
            X_train, y_train, X_test, y_test = load_cifar100_data(args.datadir)
        elif args.dataset == 'tinyimagenet':#没封装好
            X_train, y_train, X_test, y_test = load_tinyimagenet_data(args.datadir)

        n_train = y_train.shape[0]#只要y_train用到了
        np.random.seed(args.init_seed)###固定seed之后，每次划分数据集都一样
        if args.partition == "homo" or args.partition == "iid":
            idxs = np.random.permutation(n_train)
            batch_idxs = np.array_split(idxs, args.n_parties)
            net_dataidx_map = {i: batch_idxs[i] for i in range(args.n_parties)}
        elif args.partition == "noniid-labeldir" or args.partition == "noniid":
            min_size = 0
            min_require_size = 10
            K = 10
            if args.dataset == 'cifar100':
                K = 100
            elif args.dataset == 'tinyimagenet':
                K = 200
                # min_require_size = 100

            N = y_train.shape[0]
            net_dataidx_map = {}

            while min_size < min_require_size:
                idx_batch = [[] for _ in range(args.n_parties)]
                for k in range(K):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(args.beta_distribution, args.n_parties))
                    proportions = np.array(
                        [p * (len(idx_j) < N / args.n_parties) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])
                    # if K == 2 and n_parties <= 10:
                    #     if np.min(proportions) < 200:
                    #         min_size = 0
                    #         break

            for j in range(args.n_parties):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, args.logdir)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size,
                                                               args.batch_size)
        train_loaders.append(train_dl_global)#按MOON实验设置，全局模型的数据是全部数据，各个本地模型只拿到这个数据的一部分
        test_loaders.append(test_dl_global)
        sites.append("1")
        for net_id in range(args.n_parties):
            dataidxs = net_dataidx_map[net_id]
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.batch_size, dataidxs)
            train_loaders.append(train_dl_local)
            test_loaders.append(test_dl_local)
            sites.append("%d"%(net_id+2))
        return sites,trainsets, testsets,train_loaders,val_loaders,test_loaders,net_dataidx_map#trainsets, testsets,val_loaders=[]
    elif args.dataset in publish_data3:
        if args.dataset == 'lidc':
            net_dataidx_map = {}
            sites = ["0", "1", "2", "3"]
            for site in sites:
                train_set = med_DataSet(root=args.datadir, center=site, data_set="train_data",
                                        transform=transforms.ToTensor())
                test_set = med_DataSet(root=args.datadir, center=site, data_set="test_data",
                                       transform=transforms.ToTensor())
                train_dl_local = data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                                 num_workers=8, pin_memory=True)
                test_dl_local = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                                pin_memory=True)
                train_loaders.append(train_dl_local)
                test_loaders.append(test_dl_local)
                net_dataidx_map[site] = train_set
        return sites, trainsets, testsets, train_loaders, val_loaders, test_loaders, net_dataidx_map
    elif args.dataset in private_data:
        if args.dataset=='lung_nodules':
            net_dataidx_map = {}
            sites= ["江门医院","广东省医", "湛江医院", "中大五院", "中山肿瘤"]
            for site in sites:
                # train_dl_local, test_dl_local, train_ds_local, test_ds_local=get_dataloader(args.dataset, args.datadir, args.batch_size, args.batch_size,center_name=site)
                train_set=med_DataSet(root=args.datadir,center=site,data_set="train_data",transform=transforms.ToTensor())
                test_set=med_DataSet(root=args.datadir,center=site,data_set="test_data",transform=transforms.ToTensor())
                train_dl_local = data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                           num_workers=8, pin_memory=True)
                test_dl_local = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,num_workers=8,pin_memory=True)
                train_loaders.append(train_dl_local)
                test_loaders.append(test_dl_local)
                net_dataidx_map[site]=train_set
        elif args.dataset=='gastric':
            net_dataidx_map = {}
            sites=["江门医院","东莞医院", "梅州医院", "中大附一"]
            for site in sites:
                train_set = med_DataSet(root=args.datadir, center=site, data_set="train_data",
                                        transform=transforms.ToTensor())
                test_set = med_DataSet(root=args.datadir, center=site, data_set="test_data",
                                       transform=transforms.ToTensor())
                train_dl_local = data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                                 num_workers=8, pin_memory=True)
                test_dl_local = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                                pin_memory=True)
                train_loaders.append(train_dl_local)
                test_loaders.append(test_dl_local)
                net_dataidx_map[site] = train_set
        elif args.dataset=='早期肺癌':
            net_dataidx_map = {}
            sites=["江门医院","广东省医", "中大五院", "中山肿瘤"]
            for site in sites:
                train_set = med_DataSet(root=args.datadir, center=site, data_set="train_data",
                                        transform=transforms.ToTensor())
                test_set = med_DataSet(root=args.datadir, center=site, data_set="test_data",
                                       transform=transforms.ToTensor())
                train_dl_local = data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                                 num_workers=8, pin_memory=True)#args.batch_size设置的批量大小。drop_last参数设置为False，表示最后一个不完整的批次将被忽略。shuffle参数设置为True，表示在训练过程中随机打乱数据。num_workers参数设置为8，表示使用8个线程来加载数据。pin_memory参数设置为True，表示将数据加载到 CUDA 设备上
                test_dl_local = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                                pin_memory=True)
                train_loaders.append(train_dl_local)
                test_loaders.append(test_dl_local)
                net_dataidx_map[site] = train_set

        return sites, trainsets, testsets, train_loaders, val_loaders, test_loaders,net_dataidx_map  # trainsets, testsets,val_loaders=[]
if __name__ == '__main__':
    exit()



