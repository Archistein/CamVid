import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from tqdm import tqdm
from math import isclose
import segmentation_models_pytorch as smp
from typing import Optional
from dataset import *


@torch.inference_mode
def evaluate(model: nn.Module, 
             criterion: callable,
             num_classes: int,
             dataloader: data.DataLoader,
             device: torch.device
            ) -> tuple[float, float]:
    
    model.eval()

    running_loss = 0
    running_iou = 0 
    amount = 0

    for inputs, targets in (pbar := tqdm(dataloader, desc='Validation step')):
        
        inputs, targets = inputs.to(device), targets.to(device)
    
        logits = model(inputs)
        
        preds = logits.argmax(dim=1)
        loss = criterion(logits, targets)
        
        stats = smp.metrics.get_stats(preds, targets, mode='multiclass', num_classes=num_classes)
        iou_score = smp.metrics.iou_score(*stats, reduction='micro')

        running_loss += loss.item() * inputs.size(0)
        running_iou += iou_score.item() * inputs.size(0)

        amount += inputs.size(0)

        val_loss = running_loss / amount
        val_iou = running_iou / amount

        pbar.set_description(f'Val Loss: {val_loss:.06f} | Val IoU: {val_iou:.06f}')

    model.train()

    return val_loss, val_iou


@torch.no_grad
def plot_test_results(model: nn.Module, 
                      test_input: tuple[torch.Tensor, torch.Tensor],
                      device: torch.device,
                      class_to_color: dict[int, tuple[int]],
                      filename: Optional[str] = None                    
                     ) -> None:    
    model.eval()
    model.cpu()
    
    inputs, targets = test_input

    if inputs.dim() == 3:
        inputs = inputs.unsqueeze(0)
        targets = targets.unsqueeze(0)

    batch_size = inputs.size(0)
    
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    preds = model(inputs).argmax(dim=1)
    
    fig, axs = plt.subplots(nrows=batch_size, ncols=3, figsize=(12, 4 * batch_size))
    
    def get_indices(i: int, j: int) -> tuple[int, Optional[int]]:
        return (i, j) if batch_size > 1 else j

    for i in range(batch_size):
        axs[get_indices(i, 0)].set_title('Image')
        axs[get_indices(i, 0)].imshow(std * inputs[i].permute(1, 2, 0) + mean)
        axs[get_indices(i, 0)].axis("off")
        axs[get_indices(i, 1)].set_title('Ground truth')
        axs[get_indices(i, 1)].imshow(labels_to_mask(targets[i], class_to_color))
        axs[get_indices(i, 1)].axis("off")
        axs[get_indices(i, 2)].set_title('Predicted')
        axs[get_indices(i, 2)].imshow(labels_to_mask(preds[i], class_to_color))
        axs[get_indices(i, 2)].axis("off")
    
    model.train()
    model.to(device)
    
    if filename:
        plt.savefig(filename)
    
    plt.show()
    plt.close()


def trainer(model: nn.Module,
            dataloaders: dict[str, data.DataLoader],
            device: torch.device,
            num_classes: int,
            save_best: bool = False,
            epoch: int = 200,
            lr: float = 8e-4,
            alpha: float = 0.5,
            grad_clip: int = 1,
            dataset_val: Optional[data.Dataset] = None,
            class_to_color: Optional[dict[int, tuple[int]]] = None
           ) -> tuple[list[int], list[int], list[int], list[int]]:
    
    train_iou_hist, train_loss_hist = [], []
    val_iou_hist, val_loss_hist = [], []

    dice_loss = smp.losses.DiceLoss(mode='multiclass') 
    focal_loss = smp.losses.FocalLoss(mode='multiclass') 
    criterion = lambda y_pred, y_true: alpha * dice_loss(y_pred, y_true) + (1 - alpha) * focal_loss(y_pred, y_true)
    
    optimizer = optim.AdamW(model.parameters(), lr = lr)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5, min_lr=8e-5)
    
    model.train()
    model.to(device)

    last_lr = lr
    best_iou = 0

    for e in range(epoch):
        
        running_loss = 0
        running_iou = 0 
        amount = 0

        for inputs, targets in (pbar := tqdm(dataloaders['train'], desc=f'Epoch {e+1}')):

            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)
            
            preds = logits.argmax(dim=1)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            stats = smp.metrics.get_stats(preds, targets, mode='multiclass', num_classes=num_classes)
            iou_score = smp.metrics.iou_score(*stats, reduction='micro')

            running_loss += loss.item() * inputs.size(0)
            running_iou += iou_score.item() * inputs.size(0)

            amount += inputs.size(0)

            train_loss = running_loss / amount
            train_iou = running_iou / amount

            pbar.set_description(f'Epoch {e+1} | Loss: {train_loss:.06f} | IoU: {train_iou:.06f}')

        val_loss, val_iou = evaluate(model, criterion, num_classes, dataloaders['val'], device)

        scheduler.step(-val_iou)
        
        if not isclose(last_lr, scheduler.get_last_lr()[0]):
            last_lr = scheduler.get_last_lr()[0]
            tqdm.write(f'Epoch {e} | A Plateau has been reached. Reducing lr to {last_lr:.3e}')

        if val_iou > best_iou and save_best:
            best_iou = val_iou
            torch.save(model.state_dict(), f'params.pt')
            
        train_iou_hist.append(train_iou)
        train_loss_hist.append(train_loss)
        
        val_iou_hist.append(val_iou)
        val_loss_hist.append(val_loss)
        
        if dataset_val is not None:
            assert class_to_color != None, 'Class -> color map not set'
            plot_test_results(model, dataset_val[e % (len(dataset_val) - 1)], device, class_to_color)

    return train_iou_hist, train_loss_hist, val_iou_hist, val_loss_hist