import torch 
import torch.nn as nn
import torch.nn.functional as F

from .graph import GraphBuilder
from .stgcn import STGCNBlock, FullyConnectedBlock


class STGCNModel(nn.Module):

    def __init__(self, in_channels, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        self.graph = GraphBuilder(**graph_args)
        A = torch.tensor(
            self.graph.label_map, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 75
        kernel_size = (spatial_kernel_size, temporal_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.st_gcn = nn.ModuleList((
            STGCNBlock(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            STGCNBlock(64, 64, kernel_size, 1, **kwargs),
            STGCNBlock(64, 64, kernel_size, 1, **kwargs),
            STGCNBlock(64, 64, kernel_size, 1, **kwargs),
            STGCNBlock(64, 128, kernel_size, 2, **kwargs),
            STGCNBlock(128, 128, kernel_size, 1, **kwargs),
            STGCNBlock(128, 128, kernel_size, 1, **kwargs),
            STGCNBlock(128, 256, kernel_size, 2, **kwargs),
            STGCNBlock(256, 256, kernel_size, 1, **kwargs),
            STGCNBlock(256, 256, kernel_size, 1, **kwargs),
        ))

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn)

        self.fc_blocks = nn.ModuleList((
            FullyConnectedBlock(270, 256, **kwargs),
            FullyConnectedBlock(256, 128, **kwargs),
            FullyConnectedBlock(128, 64, **kwargs)
        ))

        self.output_layer = nn.Linear(in_features=64, out_features=1)
                                                                                

    def forward(self, input):
        x1, x2 = input

        # data normalization
        N, C, T, V, M = x1.size()
        x1 = x1.permute(0, 4, 3, 1, 2).contiguous()
        x1 = x1.view(N * M, V * C, T)
        x1 = self.data_bn(x1)
        x1 = x1.view(N, M, V, C, T)
        x1 = x1.permute(0, 1, 3, 4, 2).contiguous()
        x1 = x1.view(N * M, C, T, V)

        # forward pass
        for gcn, importance in zip(self.st_gcn, self.edge_importance):
            x1, _ = gcn(x1, self.A * importance)

        # global pooling
        x1 = F.avg_pool2d(x1, x1.size()[2:])
        x1 = x1.view(N, M, -1, 1, 1).mean(dim=1)

        # data concatenation
        x1 = x1.view(N, -1)
        x = torch.cat([x1, x2], dim=1)

        # fully connected blocks
        for fc_block in self.fc_blocks:
            x = fc_block(x)

        # last fully connected layer
        output = self.output_layer(x)

        return output


    def extract_feature(self, input):
        x1, x2 = input

        # data normalization
        N, C, T, V, M = x1.size()
        x1 = x1.permute(0, 4, 3, 1, 2).contiguous()
        x1 = x1.view(N * M, V * C, T)
        x1 = self.data_bn(x1)
        x1 = x1.view(N, M, V, C, T)
        x1 = x1.permute(0, 1, 3, 4, 2).contiguous()
        x1 = x1.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn, self.edge_importance):
            x1, _ = gcn(x1, self.A * importance)

        _, c, t, v = x1.size()
        features = x1.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x1 = self.fc(x1)
        output = x1.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, features