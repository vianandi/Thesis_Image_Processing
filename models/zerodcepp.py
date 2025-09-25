import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Zero-DCE++ Model Definition
# -------------------------------
class ZeroDCEPP(nn.Module):
    def __init__(self, num_layers=8, channel=32):
        super(ZeroDCEPP, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.num_layers = num_layers

        layers = []
        layers.append(nn.Conv2d(3, channel, 3, 1, 1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers-2):
            layers.append(nn.Conv2d(channel, channel, 3, 1, 1))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(channel, 24, 3, 1, 1))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x_in = x
        x = self.conv(x)

        # output 8 curve parameters
        r = torch.split(x, 3, dim=1)
        x = x_in
        for i in range(self.num_layers):
            x = x + r[i] * (torch.pow(x, 2) - x)
        return x

# -------------------------------
# Loader
# -------------------------------
def load_zerodcepp(weight_path, device="cuda"):
    model = ZeroDCEPP()
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# -------------------------------
# Apply function
# -------------------------------
def apply_zerodcepp(img_tensor, model, device="cuda"):
    """
    img_tensor: torch.Tensor [C,H,W] in range [0,1]
    """
    model.eval()
    with torch.no_grad():
        inp = img_tensor.unsqueeze(0).to(device)
        out = model(inp)
    return out.squeeze(0).clamp(0, 1).cpu()
