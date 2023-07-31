import torch
import cv2
import numpy as np
from os.path import join
from dataset import AbstractDataset_aux
from dataset import AbstractDataset
from torchvision.datasets import VisionDataset
import albumentations
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2

METHOD = ['all', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
SPLIT = ['train', 'val', 'test']
COMP2NAME = {'c0': 'raw', 'c23': 'c23', 'c40': 'c40'}
SOURCE_MAP = {'youtube': 2, 'Deepfakes': 3, 'Face2Face': 4, 'FaceSwap': 5, 'NeuralTextures': 6}

class Mydata_aux(AbstractDataset_aux):
    """
    FaceForensics++ Dataset proposed in "FaceForensics++: Learning to Detect Manipulated Facial Images"
    """

    def __init__(self, cfg, seed=2022, transforms=None, transform=None, target_transform=None):
        # pre-check
        if cfg['split'] not in SPLIT:
            raise ValueError(f"split should be one of {SPLIT}, "
                             f"but found {cfg['split']}.")
        if cfg['method'] not in METHOD:
            raise ValueError(f"method should be one of {METHOD}, "
                             f"but found {cfg['method']}.")
        if cfg['compression'] not in COMP2NAME.keys():
            raise ValueError(f"compression should be one of {COMP2NAME.keys()}, "
                             f"but found {cfg['compression']}.")
        super(Mydata_aux, self).__init__(
            cfg, seed, transforms, transform, target_transform)
        print(f"Loading data from 'FF++ {cfg['method']}' of split '{cfg['split']}' "
              f"and compression '{cfg['compression']}'\nPlease wait patiently...")

        self.categories = ['original', 'fake']
        self.targets = []
        self.aux     = []
        self.images = []
        for line in open('ff++_c40/train_aux.csv'):
            if 'fnames' in line: continue
            image = line.split()[0]
            aux   = line.split()[1]
            label = line.split()[2]
            self.images.append(image)
            self.aux.append(aux)
            self.targets.append(int(label))
        
        print("Data from 'ff++ train dataset' loaded.\n")
        print(f"Dataset contains {len(self.images)} images.\n")

class Mydata_Val_aux(AbstractDataset_aux):
    """
    FaceForensics++ Dataset proposed in "FaceForensics++: Learning to Detect Manipulated Facial Images"
    """

    def __init__(self, cfg, seed=2022, transforms=None, transform=None, target_transform=None):
        # pre-check
        if cfg['split'] not in SPLIT:
            raise ValueError(f"split should be one of {SPLIT}, "
                             f"but found {cfg['split']}.")
        if cfg['method'] not in METHOD:
            raise ValueError(f"method should be one of {METHOD}, "
                             f"but found {cfg['method']}.")
        if cfg['compression'] not in COMP2NAME.keys():
            raise ValueError(f"compression should be one of {COMP2NAME.keys()}, "
                             f"but found {cfg['compression']}.")
        super(Mydata_Val_aux, self).__init__(
            cfg, seed, transforms, transform, target_transform)
        print(f"Loading data from 'FF++ {cfg['method']}' of split '{cfg['split']}' "
              f"and compression '{cfg['compression']}'\nPlease wait patiently...")

        self.categories = ['original', 'fake']
        self.targets = []
        self.images = []
        for line in open('ff++_c40/val_aux.csv'):
            if 'fnames' in line: continue
            image = line.split()[0]
            aux   = line.split()[1]
            label = line.split()[2]
            self.images.append(image)
            self.aux.append(aux)
            self.targets.append(int(label))
        
        print("Data from 'ff++ validation dataset' loaded.\n")
        print(f"Dataset contains {len(self.images)} images.\n")

class Mydata_Test_aux(AbstractDataset_aux):
    """
    FaceForensics++ Dataset proposed in "FaceForensics++: Learning to Detect Manipulated Facial Images"
    """

    def __init__(self, cfg, seed=2022, transforms=None, transform=None, target_transform=None):
        # pre-check
        if cfg['split'] not in SPLIT:
            raise ValueError(f"split should be one of {SPLIT}, "
                             f"but found {cfg['split']}.")
        if cfg['method'] not in METHOD:
            raise ValueError(f"method should be one of {METHOD}, "
                             f"but found {cfg['method']}.")
        if cfg['compression'] not in COMP2NAME.keys():
            raise ValueError(f"compression should be one of {COMP2NAME.keys()}, "
                             f"but found {cfg['compression']}.")
        super(Mydata_Test_aux, self).__init__(
            cfg, seed, transforms, transform, target_transform)
        print(f"Loading data from 'FF++ {cfg['method']}' of split '{cfg['split']}' "
              f"and compression '{cfg['compression']}'\nPlease wait patiently...")

        self.categories = ['original', 'fake']
        self.targets = []
        self.images = []
        for line in open('ff++_c40/test_aux.csv'):
            if 'fnames' in line: continue
            image = line.split()[0]
            aux   = line.split()[1]
            label = line.split()[2]
            self.images.append(image)
            self.aux.append(aux)
            self.targets.append(int(label))
        
        print("Data from 'ff++ test dataset' loaded.\n")
        print(f"Dataset contains {len(self.images)} images.\n")


if __name__ == '__main__':
    import yaml

    config_path = "config/dataset/faceforensics.yml"
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = config["train_cfg"]
    # config = config["test_cfg"]

    def run_dataset():
        dataset = Mydata_aux(config)
        print(f"dataset: {len(dataset)}")
        for i,  in enumerate(dataset):
            path, target = _
            print(f"path: {path}, target: {target}")
            if i >= 9:
                break


    def run_dataloader(display_samples=False):
        from torch.utils import data
        import matplotlib.pyplot as plt

        dataset = Mydata_aux(config)
        dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)
        print(f"dataset: {len(dataset)}")
        for i, (images, aux, targets) in enumerate(dataloader):
            path, aux, targets = images, aux, targets
            image = dataloader.dataset.load_item(path)
            print(f"image: {image.shape}, target: {targets}")
            if display_samples:
                plt.figure()
                img = image[0].permute([1, 2, 0]).numpy()
                plt.imshow(img)
                # plt.savefig("./img_" + str(i) + ".png")
                plt.show()
            if i >= 9:
                break


    ###########################
    # run the functions below #
    ###########################

    # run_dataset()
    run_dataloader(False)

