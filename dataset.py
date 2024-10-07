import torch
from torch.utils import data
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from typing import Optional
import cv2
import os


def mask_to_labels(mask: torch.Tensor, colors: list[list[int]]) -> torch.Tensor:
    w, h = mask.shape[:2]
    labels = torch.zeros((w, h), dtype=torch.long)

    colors = torch.tensor(colors)

    for i, color in enumerate(colors):
        ind = torch.all(torch.abs((mask - color)) == 0, axis = -1)
        labels[ind] = i

    return labels


def labels_to_mask(labels: torch.Tensor, class_color_map: dict[int, tuple[int]]) -> np.ndarray:
    h, w = labels.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for label, color in class_color_map.items():
        mask[labels == label] = color
    
    return mask


class CamVid(data.Dataset):
    def __init__(self, img_root: str, 
                 mask_root: str,
                 colors: list[list[int]],
                 transforms: Optional[A.Compose] = None
                ) -> None:  
        self.img_paths = sorted([os.path.join(img_root, img) for img in os.listdir(img_root)])
        self.mask_paths = sorted([os.path.join(mask_root, mask) for mask in os.listdir(mask_root)])
        self.colors = colors
        self.transforms = transforms

        assert len(self.img_paths) == len(self.mask_paths), 'The number of images and masks are not equal'

    def __len__(self) -> int:
        return len(self.img_paths)
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            augm = self.transforms(image=image, mask=mask)
            image = augm['image']
            mask = augm['mask']
        else:
            image = torch.from_numpy(image)
            mask = torch.from_numpy(image)
            
        mask = mask_to_labels(mask, self.colors)

        return image, mask
    
    def get_transforms(infer_width: int, infer_height: int) -> dict[A.Compose]:
        return {
            'train': A.Compose([
                A.Resize(width=infer_width, height=infer_height),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.5),
                A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.AdvancedBlur(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]), 
            'val': A.Compose([
                A.Resize(width=infer_width, height=infer_height),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]),
            'test': A.Compose([
                A.Resize(width=infer_width, height=infer_height),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        }
    

def get_dataloaders(root_dir: str,
                    infer_width: int, 
                    infer_height: int,
                    colors: list[list[int]], 
                    batch_size: int,
                    get_datasets: Optional[bool] = None
                   ) -> tuple[dict[str, data.DataLoader], Optional[dict[str, data.Dataset]]]:
    transforms = CamVid.get_transforms(infer_width, infer_height)

    datasets = {x: CamVid(os.path.join(root_dir, x), os.path.join(root_dir, f'{x}_labels'), colors, transforms[x]) for x in ['train', 'val', 'test']}

    dataloaders = {x: data.DataLoader(datasets[x], batch_size=batch_size, num_workers=2) for x in ['val', 'test']}
    dataloaders['train'] = data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    if get_datasets:
        return dataloaders, datasets
    
    return dataloaders


def visualize_batch(batch: tuple[torch.Tensor, torch.Tensor], 
                    grid_size: int,
                    mean: Optional[torch.Tensor] = None,
                    std: Optional[torch.Tensor] = None) -> None:
    _, axs = plt.subplots(nrows=2, ncols=grid_size, figsize=(14, 5.5))

    inps, targets = batch

    if mean is None:
        mean = torch.tensor([0.485, 0.456, 0.406])
    if std is None:
        std = torch.tensor([0.229, 0.224, 0.225])

    for i in range(grid_size):
        inp = inps[i].permute(1, 2, 0)
        inp = std * inp + mean
        axs[0, i].imshow(inp)
        axs[0, i].axis("off")
        axs[1, i].imshow(targets[i])
        axs[1, i].axis("off")

    plt.tight_layout()
    plt.show()