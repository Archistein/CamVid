import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from typing import Optional
import timm


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.double_conv(x)

        return x


class UnetDecoder(nn.Module):
    def __init__(self, encoder_channels: list[int], decoder_channels: list[int]) -> None:
        super().__init__()

        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]

        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        features = features[::-1]  

        x = features[0]
        skips = features[1:]

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class Unet_resnet34(nn.Module):
    def __init__(self, num_classes: int, 
                 in_channels: int = 3, 
                 decoder_channels: tuple[int] = (256, 128, 64, 32, 16)
                ) -> None:
        super().__init__()
        self.encoder = timm.create_model('resnet34', 
                                         features_only=True,
                                         in_chans=in_channels,
                                         num_classes=num_classes, 
                                         pretrained=False
                                        )
        self.encoder_channels = [in_channels] + self.encoder.feature_info.channels()

        self.decoder = UnetDecoder(self.encoder_channels, decoder_channels)
        
        self.segmentation_head = nn.Conv2d(in_channels=decoder_channels[-1], out_channels=num_classes, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        return masks


def get_model(num_classes, in_channels: int = 3, params_path: Optional[str] = None) -> nn.Module:
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights=None,
        in_channels=in_channels,                  
        classes=num_classes,                  
    )

    if params_path:
        model.load_state_dict(torch.load(params_path, weights_only=True))

    return model