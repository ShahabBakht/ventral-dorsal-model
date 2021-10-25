import os, sys
from generate_SSM import *
curr_wd = os.getcwd()
sys.path.append(os.path.join(os.getcwd(),'../'))
sys.path.append('./boc/')

from allensdk.core.brain_observatory_cache import BrainObservatoryCache


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import scipy as sp
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as tmodels
from functools import partial
import collections
import pdb



PATH_visualStim = os.path.join(os.getcwd(),"../../Visual-Stimulus/StimulusImages/")

def prePareVisualStim_for_CPC(blocksize):
    
    allDGImage_paths = glob.glob(os.path.join(PATH_visualStim,"DG/*.png"))
    numDGImages = len(allDGImage_paths)
    numBlocks = int(numDGImages/blocksize)
    data = np.ndarray((1,numBlocks,1,blocksize,64,64))#scenes.shape[1],scenes.shape[1]
    f = 0
    for n in range(0,numBlocks):
        img = cv2.imread(allDGImage_paths[f],0)
        for b in range(0,blocksize):
            thisImage = np.array(img)
            thisImage = cv2.resize(thisImage,(64,64))
            
            MIN = thisImage.min()
            MAX = thisImage.max()
            thisImage = (thisImage - MIN)/MAX
            thisImage_R = (thisImage + 0.485) * 0.229
            thisImage_G = (thisImage + 0.456) * 0.224
            thisImage_B = (thisImage + 0.406) * 0.225

            data[0,n,0,b,:,:] = thisImage
            f = f + 1
    data_DG = np.concatenate((data,data,data),axis=2)
    
    
    allSGImage_paths = glob.glob(os.path.join(PATH_visualStim,"SG/*.png"))
    numSGImages = len(allSGImage_paths)
    data = np.ndarray((1,numSGImages,1,blocksize,184,184))

    for n in range(0,numSGImages):
        img = cv2.imread(allSGImage_paths[n],0)
        for b in range(0,blocksize):
            thisImage = np.array(img)
            thisImage = cv2.resize(thisImage,(184,184))

            data[0,n,0,b,:,:] = thisImage

    data_SG = np.concatenate((data,data,data),axis=2)
    
    allRDKImage_paths = glob.glob(os.path.join(PATH_visualStim,"RDK/*.png"))
    numRDKImages = len(allRDKImage_paths)
    numBlocks = int(numRDKImages/blocksize)
    data = np.ndarray((1,numBlocks,1,blocksize,64,64))
    f = 0
    for n in range(0,numBlocks):
        img = cv2.imread(allRDKImage_paths[f],0)
        for b in range(0,blocksize):
            thisImage = np.array(img)
            thisImage = cv2.resize(thisImage,(64,64))
            thisImage = (thisImage - np.mean(thisImage))/np.std(thisImage)

            data[0,n,0,b,:,:] = thisImage
            f = f + 1
    data_RDK = np.concatenate((data,data,data),axis=2)

    return data_DG, data_SG, data_RDK

def prePareAllenStim_for_CPC(exp_id,blocksize,ds_rate):
    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
    data_set = boc.get_ophys_experiment_data(exp_id)
    
    scenes = data_set.get_stimulus_template('natural_scenes')
    movie = data_set.get_stimulus_template('natural_movie_one')
    
    scenes = np.pad(scenes,((0,0),(128,128),(0,0)))

    numImages = scenes.shape[0]
    data = np.ndarray((1,numImages,3,blocksize,64,64))
    for n in range(0,numImages):
        for b in range(0,blocksize):
            thisImage = np.array(scenes[n,:,:])
            thisImage = cv2.resize(thisImage,(64,64)) #
            
            MIN = thisImage.min()
            MAX = thisImage.max()
            thisImage = (thisImage - MIN)/MAX
            thisImage_R = (thisImage + 0.485) * 0.229
            thisImage_G = (thisImage + 0.456) * 0.224
            thisImage_B = (thisImage + 0.406) * 0.225
            
            data[0,n,0,b,:,:] = thisImage_R
            data[0,n,1,b,:,:] = thisImage_G
            data[0,n,2,b,:,:] = thisImage_B
    
    data_colored = data 
    
    movie = np.pad(movie,((0,0),(152,152),(0,0)))
    movie_R = ((movie/255) - 0.485) / 0.229
    movie_G = ((movie/255) - 0.456) / 0.224
    movie_B = ((movie/255) - 0.406) / 0.225
    numFrames = movie.shape[0]
    numBlocks = int(numFrames/(ds_rate*blocksize))
    data = np.ndarray((1,numBlocks,3,blocksize,64,64))
    f = 0
    nidx = np.arange(numBlocks)
    for n in range(0,numBlocks):
        for b in range(0,blocksize):
            thisImage_R = movie_R[f,:,:]
            thisImage_R = cv2.resize(thisImage_R,(64,64))
            thisImage_G = movie_R[f,:,:]
            thisImage_G = cv2.resize(thisImage_G,(64,64))
            thisImage_B = movie_R[f,:,:]
            thisImage_B = cv2.resize(thisImage_B,(64,64))

            data[0,nidx[n],0,b,:,:] = thisImage_R
            data[0,nidx[n],1,b,:,:] = thisImage_G
            data[0,nidx[n],2,b,:,:] = thisImage_B
            
            f = f + ds_rate
            
    data2_colored = data
    print(data2_colored.shape)
    return data_colored, data2_colored

def prePareAllenStim_for_othermodels_3d(exp_id,blocksize,ds_rate):
    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
    data_set = boc.get_ophys_experiment_data(exp_id)
    
    scenes = data_set.get_stimulus_template('natural_scenes')
    movie = data_set.get_stimulus_template('natural_movie_one')
    
    scenes = np.pad(scenes,((0,0),(128,128),(0,0)))

    numImages = scenes.shape[0]
    data = np.ndarray((numImages,3,blocksize,64,64))
    for n in range(0,numImages):
        for b in range(0,blocksize):
            thisImage = np.array(scenes[n,:,:])
            thisImage = cv2.resize(thisImage,(64,64)) #
            
            MIN = thisImage.min()
            MAX = thisImage.max()
            thisImage = (thisImage - MIN)/MAX
            thisImage_R = (thisImage + 0.485) * 0.229
            thisImage_G = (thisImage + 0.456) * 0.224
            thisImage_B = (thisImage + 0.406) * 0.225
                        
            data[n,0,b,:,:] = thisImage_R
            data[n,1,b,:,:] = thisImage_G
            data[n,2,b,:,:] = thisImage_B
    
    data_colored = data
    
    movie = np.pad(movie,((0,0),(152,152),(0,0)))
    movie_R = ((movie) - 123.0) / 75.0
    movie_G = ((movie) - 123.0) / 75.0
    movie_B = ((movie) - 123.0) / 75.0
    numFrames = movie.shape[0]
    numBlocks = int(numFrames/(ds_rate*blocksize))
    data = np.ndarray((numBlocks,3,blocksize,64,64))
    f = 0
    nidx = np.arange(numBlocks)
    for n in range(0,numBlocks):
        for b in range(0,blocksize):
            
            thisImage_R = movie_R[f,:,:]
            thisImage_R = cv2.resize(thisImage_R,(64,64))
            thisImage_G = movie_R[f,:,:]
            thisImage_G = cv2.resize(thisImage_G,(64,64))
            thisImage_B = movie_R[f,:,:]
            thisImage_B = cv2.resize(thisImage_B,(64,64))

            data[nidx[n],0,b,:,:] = thisImage_R
            data[nidx[n],1,b,:,:] = thisImage_G
            data[nidx[n],2,b,:,:] = thisImage_B
            
            f = f + ds_rate
            
    data2_colored = data
    print(data2_colored.shape)
    return data_colored, data2_colored


def prePareVisualStim_for_othermodels():
    
    
    allSGImage_paths = glob.glob(os.path.join(PATH_visualStim,"SG/*.png"))
    numSGImages = len(allSGImage_paths)
    data = np.ndarray((numSGImages,1,224,224))

    for n in range(0,numSGImages):
        img = cv2.imread(allSGImage_paths[n],0)
        thisImage = np.array(img)
        thisImage = cv2.resize(thisImage,(224,224))
        data[n,0,:,:] = thisImage

    data_SG = np.concatenate((data,data,data),axis=1)
    print(data_SG.shape)

    return data_SG

def prePareAllenStim_for_othermodels(exp_id, frame_per_block=5, ds_rate=3):
    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
    data_set = boc.get_ophys_experiment_data(exp_id)
    
    scenes = data_set.get_stimulus_template('natural_scenes')
    movie = data_set.get_stimulus_template('natural_movie_one')
    
    numImages = scenes.shape[0]
    data = np.ndarray((numImages,1,224,224))
    for n in range(0,numImages):
        thisImage = np.array(scenes[n,:,0:918])
        thisImage = cv2.resize(thisImage,(224,224))
        thisImage = (thisImage - np.mean(thisImage))/np.std(thisImage)
        data[n,0,:,:] = thisImage
    
    data_colored = np.concatenate((data,data,data),axis=1)
    
    movie = np.pad(movie,((0,0),(152,152),(0,0)))
    numFrames = int(movie.shape[0] * 1/(ds_rate*frame_per_block))
    print(numFrames)
    data = np.ndarray((numFrames,1,224,224))
    f = 0
    for n in range(0,numFrames):
        thisImage = movie[f,:,:]#np.array(movie[f,:,0:303])
        thisImage = cv2.resize(thisImage,(224,224))
        thisImage = (thisImage - np.mean(thisImage))/np.std(thisImage)
        data[n,0,:,:] = thisImage
        f = f + (ds_rate * frame_per_block)
    
    data2_colored = np.concatenate((data,data,data),axis=1)
    
    
    return data_colored, data2_colored

def readAllenStimFlow_from_img(path):
    
    import numpy as np
    import cv2
    
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename))
        if img is not None:
            images.append(img)
    

    h,w,c = images[0].shape        
    I = np.empty((len(images),h,w,c))
    for n in range(len(images)):
        I[n,:,:,:] = images[n]
        
    return I

def writeAllenStim_to_videos(exp_id=501498760):
    
    import numpy as np
    import cv2
    
    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
    data_set = boc.get_ophys_experiment_data(exp_id)
    movie = data_set.get_stimulus_template('natural_movie_one')
    
    n,h,w = movie.shape
    
    # initialize water image
    height = h
    width = w
    water_depth = np.zeros((height, width), dtype=float)

    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fps = 30
    video_filename = 'output.avi'
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    for i in range(n):
        water_depth = movie[i,:,:]
        #add this array to the video
        gray = cv2.normalize(water_depth, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        gray_3c = cv2.merge([gray, gray, gray])
        out.write(gray_3c)

    # close out the video writer
    out.release()


def prepareAllenStim_for_I3D(exp_id,blocksize,dim=(64,64)):
    
    # I3D only works with videos, so here we only prepare natural video data
    
    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
    data_set = boc.get_ophys_experiment_data(exp_id)
    movie = data_set.get_stimulus_template('natural_movie_one')
    print(movie.shape)
    
    w,h = dim
    numFrames = movie.shape[0]
    numBlocks = int(numFrames/(3*blocksize))
    data = np.ndarray((1,numBlocks,1,blocksize,w,h))#scenes.shape[1],scenes.shape[1] # 184,184
    f = 0
    nidx = np.arange(numBlocks)
    for n in range(0,numBlocks):
        for b in range(0,blocksize):
            
            thisImage = np.array(movie[f,:,0:303])
            thisImage = cv2.resize(thisImage,(w,h)) #184,184
            thisImage = (thisImage - np.mean(thisImage))/np.std(thisImage)

            data[0,nidx[n],0,b,:,:] = thisImage
            f = f + 3
            
    data2_colored = np.concatenate((data,data,data),axis=2) 
    data2_colored = data2_colored.reshape((1,3,blocksize,w,h))

    return data2_colored


def get_activations_I3D(PATH,dataset,pretrained = True):
    
    curr_wd = os.getcwd()
    file_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_dir)
    sys.path.append(os.path.join(os.getcwd(),'../../pytorch-i3d/'))
    os.chdir(curr_wd)
    from pytorch_i3d import InceptionI3d
    
    load_model = PATH
    i3d = InceptionI3d(400, in_channels=2)
    i3d.replace_logits(157)
    
    if pretrained is True:
        i3d.load_state_dict(torch.load(load_model,map_location=torch.device('cpu')))
        
    for batch in dataset:   
        activations = i3d.extract_activations(batch)
        
    return activations
    
    
def get_activations_GaborPyramid3d(dataset):
    
    curr_wd = os.getcwd()
    file_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_dir)
    sys.path.append(os.path.join(os.getcwd(),'../Models/'))
    os.chdir(curr_wd) 
    
    from gabor_pyramid import GaborPyramid3d
    
    gp_model = GaborPyramid3d().to('cpu')
    
    activations = collections.defaultdict(list)

    with torch.no_grad():
        for batch in dataset:
            out = gp_model(batch)

    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    activations = {'gabor': out}

    for name in activations.keys():
        activations[name] = activations[name].detach()
    
    return activations, gp_model
                
    
def get_activations_monkeynet(PATH, dataset, pretrained = True):
    curr_wd = os.getcwd()
    file_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_dir)
    sys.path.append(os.path.join(os.getcwd(),'../Models/CPC/backbone/'))
    os.chdir(curr_wd)
    from monkeynet import SymmetricConv3d, DorsalNet
    
    model = DorsalNet(symmetric=False)
    model.eval()
    
    if pretrained is True:
        checkpoint = torch.load(PATH,map_location=torch.device('cpu'))

        pretrained_dict = {}
        for k, v in checkpoint.items():
            if 'fully_connected' not in k:
                newkey = k[7:]
                pretrained_dict[newkey] = v
        
        model_dict = model.state_dict()

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    activations = collections.defaultdict(list)
    
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d) or isinstance(m, SymmetricConv3d) or isinstance(m, nn.MaxPool3d):
            print(f'layer {name}, type {type(m)}')
                # partial to assign the layer name to each hook
                
            m.register_forward_hook(partial(save_activation, activations, name))

    with torch.no_grad():
        for batch in dataset:
            print('batch shape: ', batch.shape)
            out = model(batch)

        activations = {name: torch.cat(outputs, 0).detach() for name, outputs in activations.items()}
    for key,value in activations.items():
        activations[key] = value.detach().numpy()
            
            
    return activations, model

def get_activations_ActionRecog(PATH, dataset, pretrained = True, num_paths = 1):
    curr_wd = os.getcwd()
    file_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_dir)
    sys.path.append(os.path.join(os.getcwd(),'../Models/CPC/backbone/'))
    os.chdir(curr_wd)
    from monkeynet import SymmetricConv3d, VisualNet
    
    model = VisualNet(symmetric=True, num_res_blocks = 10, num_paths = num_paths)
    model.eval()
    
    if pretrained is True:
        checkpoint = torch.load(PATH,map_location=torch.device('cpu'))

        pretrained_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if 'linear' not in k:
                newkey = k[10:]
                pretrained_dict[newkey] = v
        
        model_dict = model.state_dict()

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    activations = collections.defaultdict(list)
    
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d) or isinstance(m, SymmetricConv3d) or isinstance(m, nn.MaxPool3d):
            print(f'layer {name}, type {type(m)}')
                # partial to assign the layer name to each hook
                
            m.register_forward_hook(partial(save_activation, activations, name))

    with torch.no_grad():
        for batch in dataset:
            print('batch shape: ', batch.shape)
            out = model(batch)

        activations = {name: torch.cat(outputs, 0).detach() for name, outputs in activations.items()}
    for key,value in activations.items():
        activations[key] = value.detach().numpy()
            
            
    return activations, model

    
def get_activations_CPC(PATH,dataset,backbone,pretrained = True):

    curr_wd = os.getcwd()
    file_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_dir)
    sys.path.append(os.path.join(os.getcwd(),'../Models/CPC/backbone/'))
    sys.path.append(os.path.join(os.getcwd(),'../Models/CPC/dpc/'))
    os.chdir(curr_wd)
    from resnet_2d3d import neq_load_customized
    from model_3d import DPC_RNN
    from convrnn import ConvGRUCell
    #from mousenet import Conv2dMask, MouseNetGRU
    from monkeynet import SymmetricConv3d

    '''
    
	PATH: path to a saved pretrained model
	batch: numpy array of images/stimuli of size (batch size X number of blocks X colors X number of frames X height X width)
	Output: a dictionary containing layer activations as tensors
    
	'''
    model = DPC_RNN(sample_size=64,#128,#48, 184
                        num_seq=60,#24,#120,#5,#8 
                        seq_len=5, 
                        network=backbone,
                        pred_step=3) #3
    all_mousenet_areas = ['LGNv',
                          'VISp4',
                          'VISp2/3',
                          'VISp5',
                          'VISal4','VISpl4','VISli4','VISrl4','VISl4',
                          'VISal2/3','VISpl2/3','VISli2/3','VISrl2/3','VISl2/3',
                          'VISal5','VISpl5','VISli5','VISrl5','VISl5',
                          'VISpor4',
                          'VISpor2/3',
                          'VISpor5']
    model.eval()
    if pretrained is True:
        checkpoint = torch.load(PATH,map_location=torch.device('cpu'))
        allkeys = list(checkpoint['state_dict'])

        pretrained_dict = {}
        for k, v in checkpoint['state_dict'].items():
            newkey = k[7:]
            pretrained_dict[newkey] = v
        
        model_dict = model.state_dict()

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    activations = collections.defaultdict(list)
	
    if backbone == 'mousenet':
        if type(model.backbone) is MouseNetGRU:
            for area in all_mousenet_areas:
                print(area)
                area_list = [area]
                [B,N,C,SL,W,H] = dataset[0].shape
                x = dataset[0].view((B*N,C,SL,W,H))
                x = x.permute(0,2,1,3,4).contiguous()
                re = model.backbone.get_img_feature_no_flatten(x,area_list)
                activations[area] = re.detach().numpy()
        else:
            for area in all_mousenet_areas:
                print(area)
                area_list = [area]
                [B,N,C,SL,W,H] = dataset[0].shape
                x = dataset[0].view((B*N,C,SL,W,H))
                x = x.permute(0,2,1,3,4).contiguous().view((B*N*SL,C,H,W)) 
                re = model.backbone.get_img_feature_no_flatten(x,area_list)
                activations[area] = re.detach().numpy()

    elif backbone == 'simmousenet':

        [B,N,C,SL,W,H] = dataset[0].shape
        x = dataset[0].view((B*N,C,SL,W,H))
        _, re = model.backbone.get_img_features(x)
        for key, value in re.items():
            if key in ['Retina','LGN','VISp_L4','VISal_L4','VISam_L4','VISl_L4','VISpm_L4']:
                BN_l4, C_l4, W_l4, H_l4 = value.shape
                value = value.view((BN_l4//SL, SL, C_l4, W_l4, H_l4))
            print(key, value.shape)
            activations[key] = value.detach().numpy()
        
        
    else:    
        weight_mean = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, ConvGRUCell) or isinstance(m, nn.MaxPool2d) or isinstance(m, SymmetricConv3d) or isinstance(m, nn.MaxPool3d):
                print(f'layer {name}, type {type(m)}')
                # partial to assign the layer name to each hook
                if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                    weight_mean.append(m.weight.mean())

                m.register_forward_hook(partial(save_activation, activations, name))
        print(f'mean weight = {sum(weight_mean)/len(weight_mean)}')
        with torch.no_grad():
            for batch in dataset:
                print('batch shape: ', batch.shape)
                out = model(batch)

            activations = {name: torch.cat(outputs, 0).detach() for name, outputs in activations.items()}
        for key,value in activations.items():
            activations[key] = value.detach().numpy()
            
            
    return activations, model


def save_activation(activations, name, mod, inp, out):#save_activation(name, mod, inp, out,activations):
    activations[name].append(out.cpu().detach())	

def get_activations_othermodels(data_,ModelName):
    
    if ModelName == 'alexnet':
        net = tmodels.alexnet(pretrained=True)
    elif ModelName == 'vgg16':
        net = tmodels.vgg16(pretrained=True)
    elif ModelName == 'resnet18':
        net = tmodels.resnet18(pretrained=True)
        
    # a dictionary that keeps saving the activations as they come
    activations = collections.defaultdict(list)
    def save_activation(name, mod, inp, out):
        activations[name].append(out.cpu())

    # Registering hooks for all the Conv2d layers
    # Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
    # called repeatedly at different stages of the forward pass (like RELUs), this will save different
    # activations. Editing the forward pass code to save activations is the way to go for these cases.
    for name, m in net.named_modules():
        if type(m)==nn.Conv2d:
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, name))

    # forward pass through the full dataset
    for batch in data_:
        out = net(batch)

    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}

    for name in activations.keys():
        activations[name] = activations[name].detach()
    
    return activations, net

def get_othermodels_RSMs(StimType,ModelName,frame_per_block=5,ds_rate=3):
    
    if StimType == 'static_gratings':
        data = prePareVisualStim_for_othermodels()
    
    elif StimType == 'natural_scenes':    
        data, _ = prePareAllenStim_for_othermodels(501498760)
        
    elif StimType == 'natural_movies':
        _, data = prePareAllenStim_for_othermodels(501498760, frame_per_block, ds_rate)
    
    dataset = [(torch.Tensor(data[:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
    activations, model = get_activations_othermodels(dataset,ModelName)
    activations_centered = center_activations(activations)
    
    all_RSM = compute_similarity_matrices(activations_centered)
    
    return all_RSM, activations, model

def get_I3D_RSMs(StimType,pretrained=False,path=''):
    
    if StimType == 'natural_movies':
        data = prepareAllenStim_for_I3D(501498760,300,dim=(256,256))
        dataset = [(torch.Tensor(data[:,:,:,:])) for _ in range(1)]
        activations = get_activations_I3D(path,dataset,pretrained = pretrained)
        activations_centered = center_activations(activations)
        all_RSM = compute_similarity_matrices(activations_centered)
        
    elif StimType == 'natural_movies_flow':
        data = prepareAllenStimFlow_for_I3D(blocksize=300,dim = (256,256))
        dataset = [(torch.Tensor(data[:,:,:,:])) for _ in range(1)]
        activations = get_activations_I3D(path,dataset,pretrained = pretrained)
        activations_centered = center_activations(activations)
        all_RSM = compute_similarity_matrices(activations_centered)

    return all_RSM, activations    
        
def get_CPC_RSMs(StimType,backbone,pretrained=False,path='',frame_per_block=5,ds_rate=3):
    

    if StimType == 'drifting_gratings':
        data_DG, _, _ = prePareVisualStim_for_CPC(5)
        dataset = [(torch.Tensor(data_DG[:,:,:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
        activations, model = get_activations_CPC(path,dataset,backbone,pretrained=pretrained)
        activations_centered = center_activations(activations)
        all_RSM = compute_similarity_matrices(activations_centered)
        del dataset
        
    elif StimType == 'static_gratings':
        _, data_SG, _ = prePareVisualStim_for_CPC(5)
        dataset = [(torch.Tensor(data_SG[:,:,:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
        activations = get_activations_CPC(path,dataset,backbone,pretrained=pretrained)
        activations_centered = center_activations(activations)
        all_RSM = compute_similarity_matrices(activations_centered)
        del dataset
    
    elif StimType == 'rdk':
        _, _, data_RDK = prePareVisualStim_for_CPC(5)
        dataset = [(torch.Tensor(data_RDK[:,:,:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
        activations, model = get_activations_CPC(path,dataset,backbone,pretrained=pretrained)
        activations_centered = center_activations(activations)
        all_RSM = compute_similarity_matrices(activations_centered)
        del dataset
    
    elif StimType == 'natural_scenes':
        data, _ = prePareAllenStim_for_CPC(501498760,frame_per_block,ds_rate)
        dataset = [(torch.Tensor(data[:,:,:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
        activations, model = get_activations_CPC(path,dataset,backbone,pretrained=pretrained)
        activations_centered = center_activations(activations)
        all_RSM = compute_similarity_matrices(activations_centered)
        del dataset
        
    elif StimType == 'natural_movies':
        _, data = prePareAllenStim_for_CPC(501498760,frame_per_block,ds_rate)
        dataset = [(torch.Tensor(data[:,:,:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
        activations, model = get_activations_CPC(path,dataset,backbone,pretrained=pretrained)
        activations_centered = center_activations(activations)
        all_RSM = compute_similarity_matrices(activations_centered)
        del dataset
   
    
    
    return all_RSM, activations, model

def get_GaborPyramid3D_RSMs(StimType,frame_per_block=5,ds_rate=3):
    
    if StimType == 'natural_movies':
        _, data2 = prePareAllenStim_for_othermodels_3d(501498760 ,frame_per_block,ds_rate)
        dataset = [(torch.Tensor(data2[:,:,:,:,:])) for _ in range(1)]
        activations, model = get_activations_GaborPyramid3d(dataset)
        activations_centered = center_activations(activations)
        all_RSM = compute_similarity_matrices(activations_centered)
        del dataset
        
    return all_RSM, activations, model

def get_monkeynet_RSMs(StimType,pretrained=True, path = '', frame_per_block=5,ds_rate=3):
    
    if StimType == 'natural_movies':
        _, data2 = prePareAllenStim_for_othermodels_3d(501498760 ,frame_per_block,ds_rate)
        dataset = [(torch.Tensor(data2[:,:,:,:,:])) for _ in range(1)]
        activations, model = get_activations_monkeynet(path, dataset, pretrained = pretrained)
        activations_centered = center_activations(activations)
        all_RSM = compute_similarity_matrices(activations_centered)
        del dataset
        
    return all_RSM, activations, model

def get_ActionRecog_RSMs(StimType,pretrained=True, path = '', frame_per_block=5,ds_rate=3, num_paths = 1):
    
    if StimType == 'natural_movies':
        _, data2 = prePareAllenStim_for_othermodels_3d(501498760 ,frame_per_block,ds_rate)
        dataset = [(torch.Tensor(data2[:,:,:,:,:])) for _ in range(1)]
        activations, model = get_activations_ActionRecog(path, dataset, pretrained = pretrained, num_paths = num_paths)
        activations_centered = center_activations(activations)
        all_RSM = compute_similarity_matrices(activations_centered)
        del dataset
        
    return all_RSM, activations, model

def get_pixel_RSMs(StimType,frame_per_block=5,ds_rate=3):
    

    if StimType == 'drifting_gratings':
        pass
        
    elif StimType == 'static_gratings':
        data = prePareVisualStim_for_othermodels()
    
    elif StimType == 'natural_scenes':    
        data, _ = prePareAllenStim_for_othermodels(501498760)
        
    elif StimType == 'natural_movies':
        _, data = prePareAllenStim_for_othermodels(501498760,frame_per_block,ds_rate)
        
    activations = {'pixel':data}
    activations_centered = center_activations(activations)
    all_RSM = compute_similarity_matrices(activations_centered)
    if StimType == 'natural_movies':
        y = activations['pixel']
        yt1 = np.reshape(y,(data.shape[0],3*224*224))
        curvature = estimate_curvature(yt1)
        
    return all_RSM, activations

def get_pixelGRU_RSMs(StimType):
    
    if StimType == 'drifting_gratings':
        data_DG, _, _ = prePareVisualStim_for_CPC(5)
        
    elif StimType == 'static_gratings':
        _, data_SG, _ = prePareVisualStim_for_CPC(5)
    
    elif StimType == 'natural_scenes':    
        data, _ = prePareAllenStim_for_CPC(501498760,5)
        
    elif StimType == 'natural_movies':
#         _, data = prePareAllenStim_for_CPC(501498760,5)
        _, data = prePareAllenStim_for_othermodels(501498760)
    
    data = torch.Tensor(np.expand_dims(data,axis=0))
    
    crnn = ConvGRU(input_size=3, hidden_size=3, kernel_size=1, num_layers=1)
    data_pixelgru, _ = crnn(data)    
    activations = {'pixel':np.squeeze(data_pixelgru.detach().numpy())}
    all_RSM = compute_similarity_matrices(activations)
    
    return all_RSM
    
def get_VGGgru_RSM(StimType):
    if StimType == 'static_gratings':
        data = prePareVisualStim_for_othermodels()
    
    elif StimType == 'natural_scenes':    
        data, _ = prePareAllenStim_for_othermodels(501498760)
        
    elif StimType == 'natural_movies':
        _, data = prePareAllenStim_for_othermodels(501498760)
    
    dataset = [(torch.Tensor(data[:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
    activations = get_activations_othermodels(dataset,"vgg16")
    
    onelayer_activation = activations[list(activations.keys())[7]]
    onelayer_activation = torch.Tensor(np.expand_dims(onelayer_activation,axis=0))

    crnn = ConvGRU(input_size=512, hidden_size=512, kernel_size=1, num_layers=1)
    activation_gru, _ = crnn(onelayer_activation)    
    vgggru_activations = {'features.17.gru':np.squeeze(activation_gru.detach().numpy())}
    all_RSM = compute_similarity_matrices(vgggru_activations)

    return all_RSM    

def get_CPC_curvature(StimType,backbone,pretrained = True,path=''):
    if StimType == 'natural_movies':
        _, data = prePareAllenStim_for_CPC(501498760,5)
        dataset = [(torch.Tensor(data[:,:,:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
        activations = get_activations_CPC(path,dataset,backbone,pretrained=pretrained)
        del dataset
        
        all_curvature = dict()
        for layer,activation_arr in activations.items():
            print(layer)
            y = activations[layer]
            y = sp.signal.resample(y,60)
            T,c,w,h = y.shape
            yt1 = np.reshape(y,(T,c*w*h))
            curvature = estimate_curvature(yt1)
            all_curvature[layer] = curvature
            
        return all_curvature

        
def estimate_curvature(y):
    # this function estimates curvature of a video sequence in a representation space (Henaf et al. Nature Neuro, 2019)
    T = y.shape[0]
    yt1 = y
    yt2 = np.roll(yt1,axis = 0,shift=1)
    vt = yt1 - yt2
    v_norm = np.ndarray((vt.shape[0],1))
    v_norm[:,0] = np.linalg.norm(vt,axis=1)
    v_norm_tile = np.tile(v_norm,(1,vt.shape[1]))
    vt_normalized = vt/v_norm_tile

    c_model = np.zeros((T,1))
    for i in range(vt_normalized.shape[0]):
        if i < vt_normalized.shape[0]-1:
            v1 = vt_normalized[i,:]
            v2 = vt_normalized[i+1,:]
            c_model[i] = np.arccos(np.inner(v1,v2))*180/np.pi
            
    return c_model
