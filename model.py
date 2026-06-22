"""
CNN Model Architecture for CIFAR-10
Upgraded to deeper 4-block ResNet-style CNN with residual connections.
Fixes:
  - set_flat_weights: use param.data.copy_() instead of direct assignment
  - Deeper architecture improves accuracy on CIFAR-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class ResidualBlock(nn.Module):
    """
    Residual Block: Conv -> BN -> ReLU -> Conv -> BN, with skip connection.
    Improves gradient flow and allows deeper networks to train effectively.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection shortcut when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)   # Residual / skip connection
        out = F.relu(out)
        return out


class CIFAR10CNN(nn.Module):
    """
    Upgraded CNN for CIFAR-10 with residual connections:
      Block 1: 3   -> 64  channels (32x32)
      Block 2: 64  -> 128 channels (16x16, stride-2)
      Block 3: 128 -> 256 channels (8x8,  stride-2)
      Block 4: 256 -> 512 channels (4x4,  stride-2)
      GlobalAvgPool -> FC(512->256) -> Dropout -> FC(256->10)

    Changes vs original:
      - 4 residual blocks instead of 3 plain conv blocks
      - No manual MaxPool; stride-2 in residual blocks handles downsampling
      - Deeper FC (512->256->num_classes)
      - Dropout reduced to 0.3 (was 0.5) for better feature retention
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super(CIFAR10CNN, self).__init__()

        # Initial stem conv (3->64, 32x32)
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Residual blocks — each stride-2 block halves spatial dims
        self.block1 = ResidualBlock(64, 64, stride=1)    # 32x32
        self.block2 = ResidualBlock(64, 128, stride=2)   # 16x16
        self.block3 = ResidualBlock(128, 256, stride=2)  # 8x8
        self.block4 = ResidualBlock(256, 512, stride=2)  # 4x4

        # Global Average Pooling: 4x4 -> 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, num_classes)

        # Weight initialisation (He init for conv, constant for BN)
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming He initialization for all conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Global average pool + flatten
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # FC classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    # ------------------------------------------------------------------
    # Weight accessors — PARAMETERS ONLY (for update delta computation)
    # ------------------------------------------------------------------

    def get_weights(self):
        """Return model parameters as a dict (excludes BN buffers)."""
        return {name: param.clone().detach() for name, param in self.named_parameters()}

    def set_weights(self, weights):
        """Set model parameters from a dict."""
        for name, param in self.named_parameters():
            if name in weights:
                param.data.copy_(weights[name])

    def get_flat_weights(self):
        """Return all *parameters* as a single flattened tensor."""
        return torch.cat([param.data.view(-1) for param in self.parameters()])

    def set_flat_weights(self, flat_weights):
        """
        Set parameters from a flattened tensor.
        FIX: Use .copy_() instead of direct assignment to preserve tensor identity.
        """
        offset = 0
        for param in self.parameters():
            numel = param.data.numel()
            param.data.copy_(flat_weights[offset:offset + numel].view(param.data.shape))
            offset += numel

    # ------------------------------------------------------------------
    # State-dict accessors — FULL state (parameters + BN buffers)
    # These MUST be used for FL aggregation to keep BN stats in sync.
    # ------------------------------------------------------------------

    def get_state_dict_flat(self):
        """
        Flatten the entire state_dict (parameters + buffers) to a single tensor.
        Critical: BN running_mean / running_var buffers are included here.
        """
        state_dict = self.state_dict()
        flat_list = []
        self._state_dict_keys = []  # Store (key, shape) for reconstruction

        for key, tensor in state_dict.items():
            self._state_dict_keys.append((key, tensor.shape))
            flat_list.append(tensor.float().view(-1))

        return torch.cat(flat_list)

    def set_state_dict_flat(self, flat_tensor):
        """
        Restore state_dict (parameters + buffers) from a flattened tensor.
        Ensures BN statistics are correctly set.
        """
        offset = 0
        new_state_dict = {}
        current_state = self.state_dict()

        for key, tensor in current_state.items():
            numel = tensor.numel()
            new_state_dict[key] = flat_tensor[offset:offset + numel].view(tensor.shape).to(tensor.dtype)
            offset += numel

        self.load_state_dict(new_state_dict)

    def get_state_dict_update(self, new_model: 'CIFAR10CNN') -> torch.Tensor:
        """
        Compute delta = new_model_flat - current_model_flat (full state_dict).
        Includes BN buffer deltas — critical for correct FL aggregation.
        """
        return new_model.get_state_dict_flat() - self.get_state_dict_flat()

    def get_weight_norm(self) -> float:
        """Debug helper: return L2 norm of all parameters."""
        total_norm = 0.0
        for param in self.parameters():
            total_norm += param.data.norm(2).item() ** 2
        return total_norm ** 0.5


def create_model(device=DEVICE):
    """Create and return a new CIFAR10CNN instance on the specified device."""
    model = CIFAR10CNN(num_classes=NUM_CLASSES)
    return model.to(device)
