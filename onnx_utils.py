import torch
import torch.nn as nn
from torch.utils import data
import onnxruntime as ort
from tqdm import tqdm
import numpy as np
from math import isclose
import segmentation_models_pytorch as smp
from train import ComboLoss
import onnx


def convert_to_onnx(model: nn.Module, dummy: torch.Tensor, device: torch.device, path: str) -> None:
    model.eval()
    model.to(device)

    torch.onnx.export(model,               
                    dummy,
                    path, 
                    export_params=True,        
                    input_names = ['input'],   
                    output_names = ['output'],
                    dynamic_axes={'input' : {0 : 'batch_size'},
                                 'output' : {0 : 'batch_size'}})
    

def check_onnx_model(torch_model: nn.Module, 
                     onnx_path: str, 
                     dummy: torch.Tensor,
                     num_classes: int,
                     dataloaders: dict[str, data.DataLoader],
                     torch_val_loss: float,
                     torch_val_iou: float,
                     alpha: float = 0.5
                    ) -> None:
    
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    torch_out = torch_model(dummy)

    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    ort_outs = ort_session.run(None, {'input': to_numpy(dummy)})

    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print('Exported model has been tested with ONNXRuntime, and the result looks good!')

    criterion = ComboLoss(alpha)

    running_loss = 0
    running_iou = 0 
    amount = 0

    for inputs, targets in (pbar := tqdm(dataloaders['val'], desc='Starting ONNX model evaluation.')):
            
        logits = ort_session.run(None, {'input': to_numpy(inputs)})[0]
        
        preds = np.argmax(logits, axis=1)
        
        loss = criterion(torch.tensor(logits), targets)
            
        stats = smp.metrics.get_stats(torch.tensor(preds), targets, mode='multiclass', num_classes=num_classes)
        iou_score = smp.metrics.iou_score(*stats, reduction='micro')

        running_loss += loss.item() * inputs.size(0)
        running_iou += iou_score.item() * inputs.size(0)

        amount += inputs.size(0)

        val_loss = running_loss / amount
        val_iou = running_iou / amount

        pbar.set_description(f'ONNX Eval | Val Loss: {val_loss:.06f} | Val IoU: {val_iou:.06f}')

    assert isclose(val_loss, torch_val_loss, rel_tol=1e-5), 'ONNX validation loss is different from torch validation loss'
    assert isclose(val_iou, torch_val_iou, rel_tol=1e-5), 'ONNX IoU is different from torch IoU'

    print('All metrics match perfectly!')