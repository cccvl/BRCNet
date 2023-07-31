import cv2
import torch
import numpy as np
from torchvision.datasets import VisionDataset
import albumentations
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2


class AbstractDataset_aux(VisionDataset):
    def __init__(self, cfg, seed=2022, transforms=None, transform=None, target_transform=None):
        super(AbstractDataset_aux, self).__init__(cfg['root'], transforms=transforms,
                                              transform=transform, target_transform=target_transform)
        # fix for re-production
        np.random.seed(seed)

        self.images  = list()
        self.aux     = list()
        self.targets = list()
        self.split = cfg['split']
        if self.transforms is None:
            self.transforms1 = Compose(
                [getattr(albumentations, _['name'])(**_['params']) for _ in cfg['transforms1']] +
                [ToTensorV2()]
            )
            self.transforms2 = Compose(
                [getattr(albumentations, _['name'])(**_['params']) for _ in cfg['transforms2']] +
                [ToTensorV2()]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        aux_path = self.aux[index]
        tgt = self.targets[index]
        return img_path, aux_path, tgt

    def load_item(self, img_items, aux_items):
        images = list()
        auxs    = list()
        for item in zip(img_items, aux_items):
            img = cv2.imread(item[0])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = self.transforms1(image=img)['image']
            
            aux = cv2.imread(item[1])
            aux = cv2.cvtColor(aux, cv2.COLOR_BGR2RGB)
            aux = self.transforms2(image=aux)['image']

            images.append(image)
            auxs.append(aux)
        return torch.stack(images, dim=0), torch.stack(auxs, dim=0)
    
