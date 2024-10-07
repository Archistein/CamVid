import torch
import argparse
import random
import os
from dataset import *
from model import get_model
from train import trainer, plot_test_results
from onnx_utils import check_onnx_model, convert_to_onnx
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--train', help='switch to the training mode', action='store_true')  
    parser.add_argument('-b', '--batch_size', help='set batch size', type=int, default=4)
    parser.add_argument('-e', '--epoch', help='set epochs number', type=int, default=200)
    parser.add_argument('-l', '--learning_rate', help='set learning rate', type=float, default=8e-4)
    parser.add_argument('-iw', '--infer_width', help='set input width', type=int, default=640)
    parser.add_argument('-ih', '--infer_height', help='set input height', type=int, default=480)
    parser.add_argument('-p', '--params_path', help='set path to pretrained params', default='params/params.pt')
    parser.add_argument('-r', '--root_dir', help='set path to data root directory', default='CamVid')  

    args = parser.parse_args()

    train_mode = args.train
    batch_size = args.batch_size
    epoch = args.epoch
    lr = args.learning_rate
    infer_width = args.infer_width
    infer_height = args.infer_height
    params_path = args.params_path
    root_dir = args.root_dir

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert os.path.exists(root_dir), f"Path '{root_dir}' doesn't exists"

    class_dict = pd.read_csv(os.path.join(root_dir, 'class_dict.csv'), index_col='name')
    class_dict = class_dict.T.to_dict('list')

    num_classes = len(class_dict)
    colors = list(class_dict.values())
    color_to_class = {tuple(color): label for label, color in enumerate(class_dict.values())}
    class_to_color = {v: k for k, v in color_to_class.items()}

    if train_mode:
        print('Train mode activated')

        dataloaders, datasets = get_dataloaders(root_dir, infer_width, infer_height, colors, batch_size, True)

        print('Visualizing a random batch sample')
        train_batch = next(iter(dataloaders['train']))
        visualize_batch(train_batch, batch_size)

        unet = get_model(num_classes)

        print('Start training')
        train_iou_hist, train_loss_hist, val_iou_hist, val_loss_hist = trainer(unet, dataloaders, device, num_classes, epoch=epoch, lr=lr, dataset_val=datasets['val'], class_to_color=class_to_color)

        print(f'Training completed successfully! Final IoU: train = {train_iou_hist[-1]:.06f}, val = {val_iou_hist[-1]:.06f}')

        print('Plotting a training history')

        plt.style.use('seaborn-v0_8-deep')

        plt.plot(train_iou_hist, label='Train')
        plt.plot(val_iou_hist, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')

        plt.grid(True)
        plt.legend()
        plt.show()

        print('Plotting inference results')

        unet.eval()
        unet.to(device)

        batch_val = next(iter(dataloaders['val']))

        plot_test_results(unet, batch_val, device, class_to_color)

        print('Convert model to ONNX')

        dummy = torch.randn(1, 3, infer_height, infer_width, requires_grad=True, device=device)
        path_to_onnx = 'onnx/unet.onnx'

        convert_to_onnx(unet, dummy, device, path_to_onnx)
        check_onnx_model(unet, path_to_onnx, dummy, num_classes, dataloaders, val_loss_hist[-1], val_iou_hist[-1])
    else:
        assert os.path.exists(params_path), f"File '{params_path}' doesn't exists"
        unet = get_model(num_classes, params_path=params_path)

    print('Inference mode')

    unet.eval()
    unet.to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)

    transforms = CamVid.get_transforms(infer_width, infer_height)['val']

    while True:
        try:
            img_path = input('Path to image: ')
        except EOFError as e:
            break

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transforms(image=image)['image'].to(device)

        with torch.no_grad():
            mask = unet(image.unsqueeze(0)).argmax(dim=1).squeeze(0)

        inp = image.permute(1, 2, 0)
        inp = std * inp + mean
        plt.imshow(inp.cpu())
        plt.imshow(labels_to_mask(mask.cpu(), class_to_color), alpha=0.6)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()