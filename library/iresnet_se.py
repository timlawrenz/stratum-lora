"""
IResNet-SE50 face recognition model — exact match for lithiumice/insightface checkpoint.

Architecture: 24 IResBlock stages (body.0-23), each with res_layer (BN-Conv-BN-Conv-BN-SE)
and optional shortcut_layer for channel changes. No PReLU.

Load with: model = IResNetSE50(); model.load_state_dict(torch.load('...pth'))
"""

import torch
import torch.nn as nn


class IResNetSE50(nn.Module):
    """IResNet-50 with SE, matching lithiumice checkpoint keys exactly."""

    def __init__(self):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),  # stride 2: 112 -> 56
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )

        # 24 body stages: channels and stride per stage
        configs = [
            # (in_ch, out_ch, stride)
            (64, 64, 1), (64, 64, 1), (64, 64, 1),     # 0-2
            (64, 128, 2), (128, 128, 1), (128, 128, 1),  # 3-5
            (128, 128, 1),                                 # 6
            (128, 256, 2), (256, 256, 1), (256, 256, 1),  # 7-9
            (256, 256, 1), (256, 256, 1), (256, 256, 1),  # 10-12
            (256, 256, 1), (256, 256, 1), (256, 256, 1),  # 13-15
            (256, 256, 1), (256, 256, 1), (256, 256, 1),  # 16-18
            (256, 256, 1), (256, 256, 1),                  # 19-20
            (256, 512, 2), (512, 512, 1), (512, 512, 1),  # 21-23
        ]

        self.body = nn.ModuleList()
        for i, (in_ch, out_ch, stride) in enumerate(configs):
            stage = nn.Module()
            stage.add_module('res_layer', _ResLayer(in_ch, out_ch, stride))
            if in_ch != out_ch:
                stage.add_module('shortcut_layer', nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                ))
            elif stride != 1:
                stage.add_module('shortcut_layer', nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                ))
            self.body.append(stage)

        # Output: BN2d -> Dropout -> Flatten -> FC -> BN1d
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.6),
            nn.Flatten(),
            nn.Linear(25088, 512),  # 512 * 7 * 7
            nn.BatchNorm1d(512),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for stage in self.body:
            out = stage.res_layer(x)
            if hasattr(stage, 'shortcut_layer'):
                x = out + stage.shortcut_layer(x)
            else:
                x = out + x
        x = self.output_layer(x)
        return x


class _ResLayer(nn.Module):
    """Residual layer matching checkpoint res_layer.0-5 naming."""

    def __init__(self, in_ch, out_ch, stride):
        super().__init__()

        # 0: BN1
        self.add_module('0', nn.BatchNorm2d(in_ch))
        # 1: Conv1
        self.add_module('1', nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False))
        # 2: BN2 (no bias in checkpoint — just gamma weight)
        bn2 = nn.BatchNorm2d(out_ch, affine=True)
        bn2.bias = None  # Remove bias to match checkpoint
        self.add_module('2', bn2)
        # 3: Conv2 (checkpoint has no bias/PReLU)
        self.add_module('3', nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False))
        # 4: BN3
        self.add_module('4', nn.BatchNorm2d(out_ch))
        # 5: SE block
        reduction = out_ch // 16
        se = nn.Sequential()
        se.add_module('fc1', nn.Conv2d(out_ch, reduction, 1, bias=False))
        se.add_module('fc2', nn.Conv2d(reduction, out_ch, 1, bias=False))
        self.add_module('5', se)

    def forward(self, x):
        m = self._modules
        out = m['0'](x)    # BN
        out = m['1'](out)  # Conv
        out = m['2'](out)  # BN
        out = m['3'](out)  # Conv
        out = m['4'](out)  # BN

        # SE: pool -> fc1 -> relu -> fc2 -> sigmoid
        se = m['5']
        w = torch.mean(out, dim=[2, 3], keepdim=True)
        w = se.fc1(w).relu()
        w = se.fc2(w).sigmoid()
        out = out * w

        return out
