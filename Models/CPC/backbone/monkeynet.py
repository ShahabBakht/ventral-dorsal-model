import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


# The initialization mean and standard deviation are estimated from ResNet-2p trained with CPC (only used for RSA)
INIT_WEIGHT_MEAN = -0.009
INIT_WEIGHT_STD = 0.152


class SymmetricConv3d(nn.Module):
    """Convolution, adding symmetric versions for equivariance."""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size,
                 stride,
                 padding,
                 weight_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_norm = weight_norm

        k = 1 / (in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2])
        w = 2 * np.sqrt(k) * (torch.rand(out_channels, in_channels, *kernel_size) - 0.5)
        w = nn.Parameter(w)
        self.register_parameter('weight', w)

    def forward(self, X):
        w = torch.cat(
            (torch.rot90(self.weight, 0, [3, 4]),
             torch.rot90(self.weight, 1, [3, 4]), 
             torch.rot90(self.weight, 2, [3, 4]), 
             torch.rot90(self.weight, 3, [3, 4])), axis=0
        )

        if self.weight_norm:
            return F.conv3d(X, 
                            w / torch.sqrt((w ** 2).sum(1, keepdims=True).sum(2, keepdims=True).sum(3, keepdims=True)), 
                            padding=self.padding, 
                            stride=self.stride)
        else:
            return F.conv3d(X, 
                            w, 
                            padding=self.padding, 
                            stride=self.stride)

class ShallowNet(nn.Module):
    """
    Create a tiny shallow network to approximate the whole freakin brain.
    """
    def __init__(self, 
                 nstartfeats=16,
                 symmetric=False,
                 dropout_rate=.1,
                 weight_norm=False
                 ):
        super().__init__()

        
        if symmetric:
            assert nstartfeats % 4 == 0
            self.conv1 = SymmetricConv3d(
                3, 
                nstartfeats // 4,
                [5, 7, 7],
                [1, 2, 2],
                padding=[2, 3, 3],
                weight_norm=weight_norm
            )
        else:
            self.conv1 = nn.Conv3d(3, 
                                nstartfeats,
                                [5, 7, 7],
                                [1, 2, 2],
                                padding=[2, 3, 3],
                                bias=False
                                )
        
            if weight_norm:
                self.conv1 = nn.utils.weight_norm(self.conv1, 'weight')

        self.bn1 = nn.BatchNorm3d(nstartfeats)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool_layer = nn.MaxPool3d(kernel_size=[1, 3, 3],
                                       stride=[1, 2, 2],
                                       padding=[0, 1, 1])

        self.dropout = nn.Dropout3d(dropout_rate)
        self._dropout_rate = dropout_rate

        self.layers = [('conv1', self.conv1), 
                       ('bn1', self.bn1), 
                       ('relu', self.relu), 
                       ('pool', self.pool_layer)]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool_layer(x)

        if self._dropout_rate > 0:
            x = self.dropout(x)

        # Endpoint
        return x

class V1Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = ShallowNet(nstartfeats=64, 
                             symmetric=True,
                             dropout_rate=0,
                             weight_norm=False)

        self.skip_conv = nn.Conv3d(64, 
                             32,
                             [1, 1, 1],
                             [1, 1, 1])
        self.bn = nn.BatchNorm3d(32)
        self.relu = nn.LeakyReLU(negative_slope=0.01)

        self.res0 = ResBlock(64, 
                             32, 
                             3, 
                             1, 
                             BottleneckTransform, 
                             8,
                             drop_connect_rate=.2)

        self.res1 = ResBlock(32, 
                             32, 
                             3, 
                             1, 
                             BottleneckTransform, 
                             8,
                             drop_connect_rate=.2)

        self.cat = Identity()
        self.dropout = nn.Dropout3d(.1)

        # Hack to get visualization working properly.
        self.layers = [('conv1', self.s1.conv1),
                       ('bn1', self.s1.bn1),
                       ('maxpool1', self.s1.pool_layer),
                       ('skip', self.skip_conv),
                       ('res0', self.res0),
                       ('res1', self.res1),
                       ('cat', self.cat),
                       ('relu', self.relu),
                       ]

    def forward(self, x):
        x0 = self.s1(x)
        skipped = self.relu(self.bn(self.skip_conv(x0)))
        x1 = self.res0(x0)
        x2 = self.res1(x1)

        x = self.cat(torch.cat([skipped, x2], dim=1))

        x = self.dropout(x)

        return x


class DorsalNet(nn.Module):
    def __init__(self, symmetric=True):
        super().__init__()
        self.s1 = ShallowNet(nstartfeats=64, 
                             symmetric=symmetric,
                             dropout_rate=0,
                             weight_norm=False)

        self.res0 = ResBlock(64, 
                             32, 
                             1, 
                             1, 
                             BottleneckTransform, 
                             8,
                             drop_connect_rate=.2)

        self.res1 = ResBlock(32, 
                             32, 
                             3, 
                             1, 
                             BottleneckTransform, 
                             8,
                             drop_connect_rate=.2)

        self.res2 = ResBlock(32, 
                             32, 
                             1, 
                             1, 
                             BottleneckTransform, 
                             8,
                             drop_connect_rate=.2)

        self.res3 = ResBlock(32, 
                             32, 
                             3, 
                             1, 
                             BottleneckTransform, 
                             8,
                             drop_connect_rate=.2)

        self.dropout = nn.Dropout3d(.1)
        self.concat = Identity()

        # Hack to get visualization working properly.
        self.layers = [('conv1', self.s1.conv1),
                       ('bn1', self.s1.bn1),
                       ('res0', self.res0),
                       ('res1', self.res1),
                       ('res2', self.res2),
                       ('res3', self.res3),
                       ('concat', self.concat),
                       ]

        self.conv1 = self.s1.conv1  # Visualize weights

    def forward(self, x):
        x0 = self.s1(x)
        x1 = self.res0(x0)
        x2 = self.res1(x1)
        x3 = self.res2(x2)
        x4 = self.res3(x3)

        # Add two types of features together
        self.concat(torch.cat((x0, x4), dim=1))

        x = self.dropout(x4)

        return x

class DorsalNet_deep(nn.Module):
    def __init__(self, symmetric=True):
        super().__init__()
        self.s1 = ShallowNet(nstartfeats=256, 
                             symmetric=symmetric,
                             dropout_rate=0,
                             weight_norm=False)

        self.res0 = ResBlock(256, 
                             128, 
                             1, 
                             1, 
                             BottleneckTransform, 
                             32,
                             drop_connect_rate=.2)

        self.res1 = ResBlock(128, 
                             128, 
                             3, 
                             1, 
                             BottleneckTransform, 
                             32,
                             drop_connect_rate=.2)

        self.res2 = ResBlock(128, 
                             128, 
                             1, 
                             1, 
                             BottleneckTransform, 
                             32,
                             drop_connect_rate=.2)

        self.res3 = ResBlock(128, 
                             128, 
                             3, 
                             1, 
                             BottleneckTransform, 
                             32,
                             drop_connect_rate=.2)
        
        self.res4 = ResBlock(128, 
                             128, 
                             1, 
                             1, 
                             BottleneckTransform, 
                             32,
                             drop_connect_rate=.2)

        self.res5 = ResBlock(128, 
                             128, 
                             3, 
                             1, 
                             BottleneckTransform, 
                             32,
                             drop_connect_rate=.2)
        
        self.res6 = ResBlock(128, 
                             128, 
                             1, 
                             1, 
                             BottleneckTransform, 
                             32,
                             drop_connect_rate=.2)

        self.res7 = ResBlock(128, 
                             128, 
                             3, 
                             1, 
                             BottleneckTransform, 
                             32,
                             drop_connect_rate=.2)
        
        self.res8 = ResBlock(128, 
                             128, 
                             1, 
                             1, 
                             BottleneckTransform, 
                             32,
                             drop_connect_rate=.2)

        self.res9 = ResBlock(128, 
                             128, 
                             3, 
                             1, 
                             BottleneckTransform, 
                             32,
                             drop_connect_rate=.2)
        

        self.dropout = nn.Dropout3d(.1)
        self.concat = Identity()

        # Hack to get visualization working properly.
        self.layers = [('conv1', self.s1.conv1),
                       ('bn1', self.s1.bn1),
                       ('res0', self.res0),
                       ('res1', self.res1),
                       ('res2', self.res2),
                       ('res3', self.res3),
                       ('res4', self.res4),
                       ('res5', self.res5),
                       ('res6', self.res6),
                       ('res7', self.res7),
                       ('res8', self.res8),
                       ('res9', self.res9),
                       ('concat', self.concat),
                       ]

        self.conv1 = self.s1.conv1  # Visualize weights

    def forward(self, x):
        x0 = self.s1(x)
        x1 = self.res0(x0)
        x2 = self.res1(x1)
        x3 = self.res2(x2)
        x4 = self.res3(x3)
        x5 = self.res4(x4)
        x6 = self.res5(x5)
        x7 = self.res6(x6)
        x8 = self.res7(x7)
        x9 = self.res8(x8)
        x10 = self.res9(x9)

        

        # Add two types of features together
        self.concat(torch.cat((x0, x10), dim=1))

        x = self.dropout(x10)

        return x
    
    
class OnePathNet(nn.Module):
    def __init__(self, num_res_blocks = 20, first_resblock_in_channels = 256):
        super().__init__()
        
        self.num_res_blocks = num_res_blocks
        self.first_resblock_in_channels = first_resblock_in_channels #128 #64
        self.resblocks_out_channels = self.first_resblock_in_channels // 2 #128 #64 #32
        self.resblocks_inner_dim = self.first_resblock_in_channels // 8 # 32 #16 #8
        
        self.res_blocks = nn.ModuleDict()
        self.res_blocks['res0'] = ResBlock(self.first_resblock_in_channels, 
                                           self.resblocks_out_channels, 
                                            1, 
                                            1, 
                                            BottleneckTransform, 
                                            self.resblocks_inner_dim,
                                            drop_connect_rate=.2)
        self.res_blocks['res1'] = ResBlock(self.resblocks_out_channels, 
                                            self.resblocks_out_channels,
                                            3, 
                                            1, 
                                            BottleneckTransform, 
                                            self.resblocks_inner_dim,
                                            drop_connect_rate=.2)
        
        for i in range(2,self.num_res_blocks,2):
            
            self.res_blocks[f'res{i}'] = ResBlock(self.resblocks_out_channels, 
                                                self.resblocks_out_channels, 
                                                 1, 
                                                 1, 
                                                 BottleneckTransform, 
                                                 self.resblocks_inner_dim,
                                                 drop_connect_rate=.2)
            
            self.res_blocks[f'res{i+1}'] = ResBlock(self.resblocks_out_channels, 
                                                     self.resblocks_out_channels, 
                                                     3, 
                                                     1, 
                                                     BottleneckTransform, 
                                                     self.resblocks_inner_dim,
                                                     drop_connect_rate=.2)
        


    def forward(self, x):
        
        for i in range(self.num_res_blocks):
            x = self.res_blocks[f'res{i}'](x)
            
        return x
    
class VisualNet(nn.Module):
    def __init__(self, symmetric=True, num_res_blocks = 20, num_paths = 1, init_weights = False):
        super().__init__()
        self.num_paths = num_paths
        first_resblock_in_channels = 256 // self.num_paths
        self.path1 = OnePathNet(num_res_blocks = num_res_blocks, first_resblock_in_channels = first_resblock_in_channels)
        if self.num_paths == 2:
            self.path2 = OnePathNet(num_res_blocks = num_res_blocks, first_resblock_in_channels = first_resblock_in_channels)

        
        self.shallow_out_channels = self.path1.first_resblock_in_channels
        
        self.s1 = ShallowNet(nstartfeats=self.shallow_out_channels, 
                             symmetric=symmetric,
                             dropout_rate=0,
                             weight_norm=False)
        
        self.dropout = nn.Dropout3d(.1)
        self.concat = Identity()
        
        if init_weights:
            self._initialize_weights()
            
            
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight,mean = INIT_WEIGHT_MEAN, std = INIT_WEIGHT_STD)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        
        x0 = self.s1(x)
        x1_1 = self.path1(x0)
        if self.num_paths == 2:
            x1_2 = self.path2(x0)
            x2 = self.concat(torch.cat((x1_1, x1_2), dim=1)) #, x1_3, x1_4
        else:
            x2 = x1_1 

        x = self.dropout(x2)
        
        return x
        
        
        

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


"""From SlowFast"""
class ResBlock(nn.Module):
    """
    Residual block.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        trans_func,
        dim_inner,
        num_groups=1,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        block_idx=0,
        drop_connect_rate=0.0,
    ):
        """
        ResBlock class constructs redisual blocks. More details can be found in:
            Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
            "Deep residual learning for image recognition."
            https://arxiv.org/abs/1512.03385
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            trans_func (string): transform function to be used to construct the
                bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            drop_connect_rate (float): basic rate at which blocks are dropped,
                linearly increases from input to output blocks.
        """
        super(ResBlock, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._drop_connect_rate = drop_connect_rate
        self._construct(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            trans_func,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
            norm_module,
            block_idx,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        trans_func,
        dim_inner,
        num_groups,
        stride_1x1,
        inplace_relu,
        dilation,
        norm_module,
        block_idx,
    ):
        # Use skip connection with projection if dim or res change.
        if (dim_in != dim_out) or (stride != 1):
            self.branch1 = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=1,
                stride=[1, stride, stride],
                padding=0,
                bias=False,
                dilation=1,
            )
            self.branch1_bn = norm_module(
                num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
            )
        self.branch2 = trans_func(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
            dilation=dilation,
            norm_module=norm_module,
            block_idx=block_idx,
        )
        # EDIT: use a leaky ReLU here.
        self.relu = nn.ReLU()

    def _drop_connect(self, x, drop_ratio):
        """Apply dropconnect to x"""
        keep_ratio = 1.0 - drop_ratio
        mask = torch.empty(
            [x.shape[0], 1, 1, 1, 1], dtype=x.dtype, device=x.device
        )
        mask.bernoulli_(keep_ratio)
        x.div_(keep_ratio)
        x.mul_(mask)
        return x

    def forward(self, x):
        f_x = self.branch2(x)
        if self.training and self._drop_connect_rate > 0.0:
            f_x = self._drop_connect(f_x, self._drop_connect_rate)
        if hasattr(self, "branch1"):
            x = self.branch1_bn(self.branch1(x)) + f_x
        else:
            x = x + f_x
        x = self.relu(x)
        return x


class BottleneckTransform(nn.Module):
    """
    Bottleneck transformation: Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        block_idx=0,
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the first
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(BottleneckTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._stride_1x1 = stride_1x1
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        dilation,
        norm_module,
    ):
        (str1x1, str3x3) = (stride, 1) if self._stride_1x1 else (1, stride)

        # Tx1x1, BN, ReLU.
        self.a = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=[self.temp_kernel_size, 1, 1],
            stride=[1, str1x1, str1x1],
            padding=[int(self.temp_kernel_size // 2), 0, 0],
            bias=False,
        )
        self.a_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x3x3, BN, ReLU.
        self.b = nn.Conv3d(
            dim_inner,
            dim_inner,
            [1, 3, 3],
            stride=[1, str3x3, str3x3],
            padding=[0, dilation, dilation],
            groups=num_groups,
            bias=False,
            dilation=[1, dilation, dilation],
        )
        self.b_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.b_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x1x1, BN.
        self.c = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.c_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.c_bn.transform_final_bn = True

    def forward(self, x):
        # Explicitly forward every layer.
        # Branch2a.
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)

        # Branch2b.
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_relu(x)

        # Branch2c
        x = self.c(x)
        x = self.c_bn(x)
        return x
    
    
if __name__ == "__main__":
    
    import time
    
    mydata = torch.FloatTensor(8, 3, 5, 64, 64).to('cuda')
    nn.init.normal_(mydata)
    
    tic = time.time()
    dorsal_net = DorsalNet().to('cuda')
    visual_net = VisualNet(num_res_blocks = 10).to('cuda')
    print(visual_net)
    
    agg_out_dorsal = dorsal_net(mydata)
    agg_out_visual = visual_net(mydata)
    
    print(agg_out_dorsal.shape, agg_out_visual.shape)

    print(time.time()-tic)
    
