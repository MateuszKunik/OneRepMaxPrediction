import torch
import torch.nn as nn


class STGCNBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dropout=0,
            residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        
        self.spatial_convolution = SpatialConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size[0]
        )

        self.temporal_convolution = TemporalConvBlock(
            channels=out_channels,
            kernel_size=kernel_size[1],
            stride=stride,
            dropout=dropout
        )

        self.residual_layer = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            residual=residual
        )
        
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, A):
        res = self.residual_layer(x)
        x, A = self.spatial_convolution(x, A)
        x = self.temporal_convolution(x) + res

        return self.relu(x), A
    

class SpatialConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        temporal_kernel_size=1,
        temporal_stride=1,
        temporal_padding=0,
        temporal_dilation=1,
        bias=True):
        super().__init__()

        self.kernel_size = kernel_size

        self.convolution_layer = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(temporal_kernel_size, 1),
            stride=(temporal_stride, 1),
            padding=(temporal_padding, 0),
            dilation=(temporal_dilation, 1),
            bias=bias
        )

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.convolution_layer(x)

        # batch_size, kernel_channels, temporal_dimension, num_nodes
        n, kc, t, v = x.size()

        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A
    

class TemporalConvBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, dropout=0):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.set_padding()

        self.temporal_convolution = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(self.kernel_size, 1),
                stride=(stride, 1),
                padding=(self.padding, 0)
            ),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout, inplace=True)
        )

    def set_padding(self):
        assert self.kernel_size % 2 == 1
        self.padding = (self.kernel_size - 1) // 2

    def forward(self, x):
        return self.temporal_convolution(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, residual=True):
        super().__init__()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.residual(x)
    

class FullyConnectedBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.fc(x)