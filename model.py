"""
model.py
--------
The Thermodynamic-Informed Neural Operator (TINO) Architecture.
Based on 1D Fourier Neural Operator (FNO).

Input: [x, HeatFlux, Pressure, InletTemp, MassFlow] (5 Channels)
Output: [WallTemperature] (1 Channel)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        modes: Number of Fourier modes to keep (Frequency cutoff)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # 1. Compute Fourier coefficients (FFT)
        x_ft = torch.fft.rfft(x)

        # 2. Filter high frequencies (Keep only 'modes' lowest freqs)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights1)

        # 3. Inverse FFT back to physical domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class TINO_FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(TINO_FNO1d, self).__init__()
        """
        The Full Network.
        modes: Number of Fourier modes (e.g., 16)
        width: Number of channels in the hidden layers (e.g., 64)
        """
        self.modes = modes
        self.width = width
        
        # Input Channel (5 vars: x, q, P, T, mdot) -> Lift to 'width' channels
        self.p = nn.Linear(5, self.width) 
        
        # Fourier Layers (The "Operator" Block)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)
        
        # Skip Connections (ResNet style)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        # Output Projection
        self.q = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1) # Output: Temperature

    def forward(self, x):
        # x shape: [Batch, Grid, 5]
        
        # 1. Lift to higher dimension
        x = self.p(x) # [Batch, Grid, Width]
        x = x.permute(0, 2, 1) # [Batch, Width, Grid] for Conv1d

        # 2. Fourier Layers
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # 3. Project back to output
        x = x.permute(0, 2, 1) # [Batch, Grid, Width]
        x = self.q(x)
        x = F.gelu(x)
        x = self.fc2(x) # [Batch, Grid, 1]
        
        return x.squeeze(-1)