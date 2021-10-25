import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import networkx as nx
import yaml

from SimMouseNet_network import Network
from convrnn import ConvGRU
from LGN import Conv3dLGN_layer, Conv3dLGN



class SimMouseNet(nn.Module):
    def __init__(self,init_weights=True, hyperparam_loc='./SimMouseNet_hyperparam.yaml'):
        super(SimMouseNet, self).__init__()
        
        self.MouseGraph = Network()
        self.MouseGraph.create_graph()
        
        self.Areas = nn.ModuleDict()
        
        self.hyperparams = self.hyperparams = yaml.load(open('./SimMouseNet_hyperparams.yaml'), Loader=yaml.FullLoader)


        
        self.make_SimMouseNet()
        
        if init_weights:
            self._initialize_weights()
        
    
    def make_SimMouseNet(self):
        
#         self.Aeas['Retina'] = Retina()
#         self.Areas['LGN'] = LGN()
        
        self.AREAS_LIST = self.MouseGraph.G.nodes
        self.OUTPUT_AREA_LIST = self.MouseGraph.OUTPUT_AREA_LIST
        
        for area in self.AREAS_LIST:
            hyperparams = self.hyperparams['model'][area]

            if area == 'Retina':
                
                self.Areas[area] = Retina(in_channels = hyperparams['in_channels'], 
                                          out_channels = hyperparams['out_channels'], 
                                          kernel_size = hyperparams['kernel_size'], 
                                          padding = hyperparams['padding'])
            
            elif area == 'LGN':
                
                predec_area = list(self.MouseGraph.G.predecessors(area))
                
                this_area_in_channels = self.hyperparams['model'][predec_area[0]]['out_channels']
                
                self.Areas[area] = LGN(in_channels = this_area_in_channels, 
                                       out_channels = hyperparams['out_channels'], 
                                       kernel_size = hyperparams['kernel_size'], 
                                       padding = hyperparams['padding'])
            

            else:
                predec_area = list(self.MouseGraph.G.predecessors(area))
                
                if area == 'VISp':
                
                    this_area_in_channels = self.hyperparams['model'][predec_area[0]]['out_channels']
                
                else:
                    this_area_in_channels = self.hyperparams['model'][predec_area[0]]['L5_hidden_size'] #self.hyperparams['model'][predec_area[0]]['L4_out_channels']
                    
                    
                self.Areas[area] = Area(L4_in_channels = this_area_in_channels, 
                                         L4_out_channels = hyperparams['L4_out_channels'], 
                                         L4_kernel_size = hyperparams['L4_kernel_size'], 
                                         L4_padding = hyperparams['L4_padding'],
                                         L2_3_kernel_size = hyperparams['L2_3_kernel_size'], 
                                         L2_3_stride = hyperparams['L2_3_stride'], 
                                         L5_kernel_size = hyperparams['L5_kernel_size'], 
                                         L5_hidden_size = hyperparams['L5_hidden_size'])
                
                    
                

    def forward(self, input):
        
        Out2Agg, _ = self.get_img_features(input)
        
        return Out2Agg 
    
    def get_img_features(self, input):
        
        BN, C, SL, W, H = input.shape
        input_tempflat = input.permute(0,2,1,3,4).contiguous().view((BN*SL,C,H,W)) 

        Out = dict()
        for area in self.AREAS_LIST:

            if area == 'Retina':
                Out[area] = self.Areas[area](input_tempflat, SL)
                
            elif area == 'LGN':
                
                Out[area] = self.Areas[area](Out['Retina'])
                
            else:
                predec_area = list(self.MouseGraph.G.predecessors(area))
                
                if area == 'VISp':
#                     ipdb.set_trace()
                    this_Out = self.Areas[area](Out[predec_area[0]],SL)
    
                    Out[area+'_L4'] = this_Out[2]
                    Out[area+'_L2_3'] = this_Out[1]
                    Out[area+'_L5'] = this_Out[0]
                    BN, SL, C_, W_, H_ = Out[area+'_L5'].shape
#                     print(BN, SL, C_, W_, H_)
                    if area in self.OUTPUT_AREA_LIST:
                        Out_to_agg = torch.nn.functional.avg_pool2d(Out[area+'_L5'].view((BN*SL, C_, W_, H_)), kernel_size=2, stride=2).contiguous().view((BN,SL,C_, W_//2, H_//2)).contiguous()
                else:
                    L2_3_out_shape = Out[predec_area[0]+'_L2_3'].shape
                    this_Out = self.Areas[area](Out[predec_area[0]+'_L2_3'].view((L2_3_out_shape[0]*L2_3_out_shape[1],L2_3_out_shape[2],L2_3_out_shape[3],L2_3_out_shape[4])),SL)
#                     this_Out = self.Areas[area](Out[predec_area[0]+'_L2_3'],SL)
                    
                    Out[area+'_L4'] = this_Out[2]
                    Out[area+'_L2_3'] = this_Out[1]
                    Out[area+'_L5'] = this_Out[0]
                    
                    if area in self.OUTPUT_AREA_LIST:
                        try:
                            Out_to_agg = torch.cat((Out_to_agg,Out[area+'_L5']),dim=2)
                        except:
                            Out_to_agg = Out[area+'_L5']

        
        Out2Agg = Out_to_agg.permute(0,2,1,3,4).contiguous()
        
        return Out2Agg, Out
    
                
                
    def _initialize_weights(self):
        for m in self.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d)) and not isinstance(m, Conv3dLGN):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Area(nn.Module):
    def __init__(self, 
                    L4_in_channels, L4_out_channels, L4_kernel_size, L4_padding,
                    L2_3_kernel_size, L2_3_stride, 
                    L5_kernel_size, L5_hidden_size,
                ):
        super(Area, self).__init__()

        self.L4 = Layer_4(in_channels = L4_in_channels, out_channels = L4_out_channels, kernel_size = L4_kernel_size, padding = L4_padding)
        self.L2_3 = Layer_2_3(kernel_size = L2_3_kernel_size, stride = L2_3_stride)
        self.L5 = Layer_5(input_size = 1*L4_out_channels, kernel_size = L5_kernel_size, hidden_size = L5_hidden_size)


    def forward(self, input,frames_per_block):
        
        
        out_l4 = self.L4(input)
#         out_l2_3 = self.L2_3(out_l4)
        
        # to concatenate l4 and l2_3 output to feed to l5
#         out_l4_to_l5 = nn.functional.avg_pool2d(out_l4, kernel_size=out_l4.shape[2]//out_l2_3.shape[2], stride=out_l4.shape[2]//out_l2_3.shape[2])

        BNSL, C_l45, W_l45, H_l45 = out_l4.shape
        out_l4_to_l5 = out_l4.view((BNSL//frames_per_block,frames_per_block, C_l45, W_l45, H_l45)).contiguous()
        #out_l4_to_l5 = out_l4_to_l5.view((frames_per_block, BNSL//frames_per_block, C_l45, W_l45, H_l45)).contiguous().permute((1,0,2,3,4)).contiguous()

#         BNSL, C_l2_3, W_l2_3, H_l2_3 = out_l2_3.shape
#         out_l2_3_to_l5 = out_l2_3.view((BNSL//frames_per_block, frames_per_block, C_l2_3, W_l2_3, H_l2_3)).contiguous()
        #out_l2_3_to_l5 = out_l2_3.view((frames_per_block, BNSL//frames_per_block, C_l2_3, W_l2_3, H_l2_3)).contiguous().permute((1,0,2,3,4)).contiguous()
        
#         in_l5 = torch.cat((out_l4_to_l5,out_l2_3_to_l5),dim=2)
        in_l5 = out_l4_to_l5
        out_l5 = self.L5(in_l5)
        out_l2_3 = out_l5 #self.L2_3(out_l5)
#         print(out_l5.shape)
        return out_l5, out_l2_3, out_l4

        

class LGN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        super(LGN, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding)
        self.nonlnr = nn.ReLU()
        self.bn= nn.BatchNorm2d(out_channels)
    
    def forward(self, input):

        lgn_out = self.conv(input)
        lgn_out = self.bn(lgn_out)
        lgn_out = self.nonlnr(lgn_out)
        
        return lgn_out

class Retina(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        super(Retina, self).__init__()

#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = kernel_size, padding = padding)
#         self.conv = Conv3dLGN_layer(in_channels = in_channels, kernel_size = (kernel_size,kernel_size,kernel_size))
        self.nonlnr = nn.ReLU()
        self.bn= nn.BatchNorm3d(out_channels)
    
    def forward(self, input, frames_per_block):

        if isinstance(self.conv, nn.Conv3d) or isinstance(self.conv, Conv3dLGN_layer):
            BNSL,C,H,W = input.shape
            input = input.view((BNSL//frames_per_block,frames_per_block,C,H,W)).contiguous().permute(0,2,1,3,4)
            
        retina_out = self.conv(input)
        retina_out = self.bn(retina_out)
        retina_out = self.nonlnr(retina_out)
        if isinstance(self.conv, nn.Conv3d) or isinstance(self.conv, Conv3dLGN_layer):
            retina_out = retina_out.permute(0,2,1,3,4).contiguous().view((BNSL,self.conv.out_channels,H,W)).contiguous()

        return retina_out


class Layer_4(nn.Module):
    scale = 4
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        super(Layer_4, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels * self.scale, kernel_size = 1)
        self.bn2 = nn.BatchNorm2d(out_channels * self.scale)
        self.conv_3 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale, kernel_size = kernel_size, padding = padding, stride = 2) # if maxpool at the output of L4, strid --> 1
        self.bn3 = nn.BatchNorm2d(out_channels * self.scale)
        self.conv_4 = nn.Conv2d(out_channels * self.scale, out_channels, kernel_size = 1)
#         self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding)
#         self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, padding = padding)
#         self.conv_3 = nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, padding = padding)
#         self.conv_4 = nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, padding = padding)
        self.nonlnr = nn.ReLU()
        self.bn4= nn.BatchNorm2d(out_channels)
        
        # adding maxpool to L4 output
#         self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, input):

        l4_out_1 = self.conv_1(input)
        l4_out = self.bn1(l4_out_1)
        l4_out = self.nonlnr(l4_out)
        l4_out_2 = self.conv_2(l4_out)
        l4_out = self.bn2(l4_out_2)
        l4_out = self.nonlnr(l4_out)
        l4_out_3 = self.conv_3(l4_out)
        l4_out = self.bn3(l4_out_3)
        l4_out = self.nonlnr(l4_out)
        l4_out_4 = self.conv_4(l4_out)
        l4_out = self.bn4(l4_out_4)
        l4_out = self.nonlnr(l4_out)
        
        # passing L4 output through a maxpool
#         l4_out = self.maxpool(l4_out)
        
        return l4_out

class Layer_2_3(nn.Module):
    def __init__(self, kernel_size = 2, stride = 2):
        super(Layer_2_3, self).__init__()

#         self.maxpool = nn.MaxPool2d(kernel_size = kernel_size, stride = stride)

    def forward(self, input):
        B, N, C, W, H = input.shape
        l2_3_out = input.view((B*N,C,W,H))
#         l2_3_out = self.maxpool(input)
        
        return l2_3_out

class Layer_5(nn.Module):
    def __init__(self, input_size, kernel_size, hidden_size = 20):
        super(Layer_5, self).__init__()
        
        self.convgru = ConvGRU(input_size = input_size, hidden_size = hidden_size, kernel_size=kernel_size, num_layers=1)
#         self.conv_1 = nn.Conv2d(in_channels = input_size, out_channels = hidden_size, kernel_size = kernel_size, padding = kernel_size//2)
#         self.conv_2 = nn.Conv2d(in_channels = hidden_size, out_channels = hidden_size, kernel_size = kernel_size, padding = kernel_size//2)
#         self.conv_3 = nn.Conv2d(in_channels = hidden_size, out_channels = hidden_size, kernel_size = kernel_size, padding = kernel_size//2)
#         self.conv_4 = nn.Conv2d(in_channels = hidden_size, out_channels = hidden_size, kernel_size = kernel_size, padding = kernel_size//2)
#         self.nonlnr = nn.ReLU()
#         self.bn= nn.BatchNorm2d(hidden_size) 

    def forward(self, input):
        
#         B,N,C,W,H = input.shape
#         input = input.view((B*N, self.conv_1.in_channels, W, H)).contiguous()
        l5_out, _ = self.convgru(input)
#         l5_out = self.conv_1(input)
#         l5_out = self.bn(l5_out)
#         l5_out = self.nonlnr(l5_out)
#         l5_out = self.conv_2(l5_out)
#         l5_out = self.bn(l5_out)
#         l5_out = self.nonlnr(l5_out)
#         l5_out = self.conv_3(l5_out)
#         l5_out = self.bn(l5_out)
#         l5_out = self.nonlnr(l5_out)
#         l5_out = self.conv_4(l5_out)
#         l5_out = self.bn(l5_out)
#         l5_out = self.nonlnr(l5_out)
#         l5_out = l5_out.view((B, N, self.conv_4.out_channels, W, H)).contiguous()
        
        return l5_out
        
        
if __name__ == '__main__':
    
    import time
#     import ipdb
    
    mydata = torch.FloatTensor(10, 3, 11, 64, 64).to('cuda')
    nn.init.normal_(mydata)
    
    tic = time.time()
    sim_mouse_net = SimMouseNet(hyperparam_loc='./SimMouseNet_hyperparams.yaml').to('cuda')
        
    agg_out = sim_mouse_net(mydata)

    print(time.time()-tic)
#     ipdb.set_trace()

