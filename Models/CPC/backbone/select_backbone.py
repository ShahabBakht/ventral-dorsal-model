import os
import sys
import torch
import torch.nn as nn
from resnet_2d3d import * 
from vgg_2d import *
from SimMouseNet import *
from monkeynet import *

try:
    sys.path.append('../../Mouse_CNN/cmouse')
    import network
    import mousenet
except:
    print('MouseNet does not exist')

def select_resnet(network, track_running_stats=True,):
    param = {'feature_size': 1024}
    if network == 'resnet18':
        model = resnet18_2d3d_full(track_running_stats=track_running_stats)
        param['feature_size'] = 256
    elif network == 'resnet34':
        model = resnet34_2d3d_full(track_running_stats=track_running_stats)
        param['feature_size'] = 256 
    elif network == 'resnet50':
        model = resnet50_2d3d_full(track_running_stats=track_running_stats)
    elif network == 'resnet101':
        model = resnet101_2d3d_full(track_running_stats=track_running_stats)
    elif network == 'resnet152':
        model = resnet152_2d3d_full(track_running_stats=track_running_stats)
    elif network == 'resnet200':
        model = resnet200_2d3d_full(track_running_stats=track_running_stats)
    elif network == 'resnet0':
        model = resnet_basic_2d3d_full(track_running_stats=track_running_stats)
        param['feature_size'] = 256
    elif network == 'vgg':
        model = vgg16_bn(pretrained=False) #vgg16_bn
        #model = vgg8(pretrained=False)
        param['feature_size'] = 512
    else: raise IOError('model type is wrong')

    return model, param

def select_mousenet():
    param = {'feature_size': None}
    net = network.load_network_from_pickle('/Users/shahab/Mila/Project-Codes/Mouse_CNN/example/network_(3,64,64).pkl')
    model = mousenet.MouseNet(net) #MouseNetGRU
    param['output_area_list'] = ['VISl5', 'VISrl5', 'VISli5', 'VISpl5', 'VISal5', 'VISpor5']#'VISp5', 
#     param['output_area_list'] = ['VISp5']
    all_area_list = model.network.area_channels
    
    feature_size = 0
    for area in param['output_area_list']:
        if area == 'VISp5':
            feature_size = feature_size + 32 #(32 for all readout, 40 for VISp5 readout)
        else:
            feature_size = feature_size + all_area_list[area]
    
    param['feature_size'] = int(feature_size) 
    
    return model, param

def select_simmousenet():
    param = {'feature_size': None}
    model = SimMouseNet()
    param['output_area_list'] = model.OUTPUT_AREA_LIST
    
    param['feature_size'] =  256 #, 56 #256 # #256 

    # TO DO: make feature_size calculation automatic
    
    return model, param

def select_monkeynet(num_paths = 1):
    param = {'feature_size': None}
#     model = DorsalNet_deep()
    model = VisualNet(num_res_blocks=10, num_paths = num_paths, init_weights = True)
    param['feature_size'] = num_paths * model.path1.resblocks_out_channels #96 #160 #160 #256 #
    
    return model, param