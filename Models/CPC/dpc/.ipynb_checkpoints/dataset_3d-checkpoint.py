import torch
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import CIFAR10
import os
import sys
import time
import pickle
import glob
import csv
import pandas as pd
import numpy as np
import cv2
sys.path.append('../utils')

from augmentation import *
from tqdm import tqdm
from joblib import Parallel, delayed
import collections
import datetime
import hashlib
import matplotlib
import matplotlib.image
import pandas as pd
from pathlib import Path
import requests
import struct
import subprocess
import tables
import h5py
from PIL import Image
import tempfile


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class Kinetics400_full_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 epsilon=5,
                 unit_test=False,
                 big=False,
                 return_label=False):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.unit_test = unit_test
        self.return_label = return_label

        if big: print('Using Kinetics400 full data (256x256)')
        else: print('Using Kinetics400 full data (150x150)')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        action_file = os.path.join('../process_data/data/kinetics400', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=',', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1 # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # splits
        if big:
            if mode == 'train':
                split = '../process_data/data/kinetics400_256/train_split.csv'
                video_info = pd.read_csv(split, header=None)
            elif (mode == 'val') or (mode == 'test'):
                split = '../process_data/data/kinetics400_256/val_split.csv'
                video_info = pd.read_csv(split, header=None)
            else: raise ValueError('wrong mode')
        else: # small
            if mode == 'train':
                split = '../process_data/data/kinetics400/train_split.csv'
                video_info = pd.read_csv(split, header=None)
            elif (mode == 'val') or (mode == 'test'):
                split = '../process_data/data/kinetics400/val_split.csv'
                video_info = pd.read_csv(split, header=None)
            else: raise ValueError('wrong mode')

        drop_idx = []
        print('filter out too short videos ...')
        for idx, row in tqdm(video_info.iterrows(), total=len(video_info)):
            vpath, vlen = row
            if vlen-self.num_seq*self.seq_len*self.downsample <= 0:
                drop_idx.append(idx) 
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val': self.video_info = self.video_info.sample(frac=0.3, random_state=666)
        if self.unit_test: self.video_info = self.video_info.sample(32, random_state=666)
        # shuffle not necessary because use RandomSampler

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen-self.num_seq*self.seq_len*self.downsample <= 0: return [None]
        n = 1
        start_idx = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), n)
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1)*self.downsample*self.seq_len + start_idx
        seq_idx_block = seq_idx + np.expand_dims(np.arange(self.seq_len),0)*self.downsample
        return [seq_idx_block, vpath]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        if items is None: print(vpath) 
        
        idx_block, vpath = items
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq*self.seq_len)
        
        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
        t_seq = self.transform(seq) # apply same transform
        
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)

            label = torch.LongTensor([vid])
            return t_seq, label

        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


class CatCam_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None, 
                 seq_len=10,
                 num_seq = 5,
                 downsample=3,
                 epsilon=5,
                 which_split=1,
                 return_label=False):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.which_split = which_split
        self.return_label = return_label

        # splits
        if mode == 'train':
            split = '../process_data/data/catcam/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'): # use val for test
            split = '../process_data/data/catcam/test_split%02d.csv' % self.which_split 
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        action_file = os.path.join('../process_data/data/catcam', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen-self.num_seq*self.seq_len*self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val': self.video_info = self.video_info.sample(frac=0.3)
        # shuffle not required due to external sampler

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen-self.num_seq*self.seq_len*self.downsample <= 0: return [None]
        n = 1
        start_idx = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), n)
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1)*self.downsample*self.seq_len + start_idx
        seq_idx_block = seq_idx + np.expand_dims(np.arange(self.seq_len),0)*self.downsample
        return [seq_idx_block, vpath]


    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        if items is None: print(vpath) 
        
        idx_block, vpath = items
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq*self.seq_len)
        
        seq = [pil_loader(os.path.join(vpath, 'Catt%04d.tif' % (i+1))) for i in idx_block]
        t_seq = self.transform(seq) # apply same transform
        
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)
            label = torch.LongTensor([vid])
            return t_seq, label
            
        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return action code'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]

class UCF101_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None, 
                 seq_len=10,
                 num_seq = 5,
                 downsample=3,
                 epsilon=5,
                 which_split=1,
                 return_label=False):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.which_split = which_split
        self.return_label = return_label

        # splits
        if mode == 'train':
            split = '../process_data/data/ucf101/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'): # use val for test
            split = '../process_data/data/ucf101/test_split%02d.csv' % self.which_split 
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        action_file = os.path.join('../process_data/data/ucf101', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen-self.num_seq*self.seq_len*self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val': self.video_info = self.video_info.sample(frac=0.3)
        # shuffle not required due to external sampler

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen-self.num_seq*self.seq_len*self.downsample <= 0: return [None]
        n = 1
        start_idx = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), n)
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1)*self.downsample*self.seq_len + start_idx
        seq_idx_block = seq_idx + np.expand_dims(np.arange(self.seq_len),0)*self.downsample
        return [seq_idx_block, vpath]


    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        if items is None: print(vpath) 
        
        idx_block, vpath = items
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq*self.seq_len)
        
        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
        t_seq = self.transform(seq) # apply same transform
        
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)
            label = torch.LongTensor([vid])
            return t_seq, label

        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return action code'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]

cache = {10: {}, 40: {}}

nclasses = 72  # 5 degree precision in heading discrimination.
max_speed = 3  # Max 3 m/s movement


def to_class(theta):
    theta = theta % (2 * np.pi)
    return int(theta / (2 * np.pi) * nclasses)


def to_linear_class(speed, maxspeed):
    return int(speed / maxspeed * nclasses)


class AirSim(data.Dataset):
    """
    Loads a segment from the Airsim flythrough data.
    """

    def __init__(self, root="./airsim", split="train", regression=True, nt=40, seq_len=5, num_seq=8, transform = None, return_label=False):

        if split not in ("train", "tune", "val", "report", "traintune"):
            raise NotImplementedError("Split is set to an unknown value")

        assert nt in (10, 40)
        assert nt == seq_len * num_seq
        
        self.return_label = return_label
        self.split = split
        self.root = root
        
        cells = []
        for item in Path(root).glob("*/*/*.h5"):
            if ('output.h5' in str(item)) and ("2021-02-03T035302" not in str(item)) and ("2021-02-04T104447" not in str(item)) and ("mountains" not in str(item)):
                cells.append(item)

        cells = sorted(cells)
            
        splits = {
            "train": [0, 1, 2, 3, 5, 6, 7, 8],
            "tune": [4],
            "val": [4],
            "report": [9],
            "traintune": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
        nblocks = 10
        sequence = []
        i = 0
        for cell in cells:
            f = tables.open_file(cell, "r")
            labels = f.get_node("/labels")[:]
            f.close()
            cell_path, _ = os.path.split(cell)
            for j in range(labels.shape[0]):
                tensor_path = os.path.join(cell_path,'split',f'{j}.h5')            
                if (j % nblocks) in splits[split]:
                    if regression:
                        # Outputs appropriate for regression
                        sequence.append(
                            {
                                "images_path": cell,
                                "tensor_path": tensor_path,
                                "labels": np.array(
                                        [
                                            labels[j]["heading_pitch"],
                                            labels[j]["heading_yaw"],
                                            labels[j]["rotation_pitch"],
                                            labels[j]["rotation_yaw"],
                                            labels[j]["speed"],
                                        ],
                                        dtype=np.float32,
                                    ),
                                "idx": j,
                            }
                        )
                    else:
                        # Outputs appropriate for multi-class
                        hp = to_class(labels[j]["heading_pitch"])
                        hy = to_class(labels[j]["heading_yaw"])
                        rp = to_class(labels[j]["rotation_pitch"])
                        ry = to_class(labels[j]["rotation_yaw"])

                            # TODO(pmin): make max_speed not hard-coded.
                        speed = to_linear_class(labels[i]["speed"], max_speed)

                        sequence.append(
                                {
                                    "images_path": cell,
                                    "labels": np.array(
                                        [hp, hy, rp, ry, speed], dtype=np.int64
                                    ),  # Torch requires long ints
                                    "idx": j,
                                }
                            )
                        
        if regression:
            self.noutputs = 5
            self.nclasses = 1
        else:
            self.noutputs = 5
            self.nclasses = nclasses

        self.sequence = sequence
        self.nt = nt
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.transform = transform
        
        if len(self.sequence) == 0:
            raise Exception("Didn't find any data")

    def __getitem__(self, idx):
        # Load a single segment of length idx from disk.
#         global cache
        tgt = self.sequence[idx]
        tensor_path = tgt["tensor_path"]
        f = tables.open_file(tensor_path, "r")
        X_ = f.get_node("/videos")[:].squeeze()
        f.close()
#         return (X_.transpose((1, 0, 2, 3)), tgt["labels"])
#         if tgt["images_path"] not in cache:
#             f = tables.open_file(tgt["images_path"], "r")
#             if self.nt == 40:
#                 X_ = f.get_node("/videos")[:].squeeze()
#             else:
#                 X_ = f.get_node("/short_videos")[:].squeeze()

#             f.close()

#             cache[self.nt][tgt["images_path"]] = X_
            
#         X_ = cache[self.nt][tgt["images_path"]]
#         X_ = X_[tgt["idx"], :].astype(np.uint8)

#         return (X_.transpose((1, 0, 2, 3)), tgt["labels"])
        seq = [Image.fromarray(np.uint8(X_[i,:,:,:]).transpose(1,2,0)) for i in range(self.nt)]
        t_seq = self.transform(seq)
        del X_, seq
        t_seq = torch.stack(t_seq, 0)
        
        t_seq = t_seq.view(self.num_seq, self.seq_len, *t_seq.shape[1:4]).transpose(1,2)
        
#         # The images are natively different sizes, grayscale.
        if self.return_label:
            return (t_seq, tgt["labels"])
        
        return t_seq

    def __len__(self):
        # Returns the length of a dataset
        return len(self.sequence)
    
class RandomDots(data.Dataset):
    def __init__(self, root="/Tmp/slurm.827552.0", split="train", regression=True, nt=40, seq_len=5, num_seq=8, transform = None, return_label=False, fine_classification = True):
        super().__init__()
        
        self.return_label = return_label
        # make a list of folders in each condition - each folder is one sample
        cells = []
        if split == "train":
            for item in Path(root).glob("train/smpl*/*/"):
                cells.append(item)
        elif split == "val":
            for item in Path(root).glob("val/smpl*/*/"):
                cells.append(item)
                
        cells = sorted(cells)
        
        sequence = []
        i = 0
        for cell in cells:
            
            # create labels
            if 'cmplx_' in str(cell):
                label_glob = 1
            elif 'smpl_' in str(cell):
                label_glob = 0
            if '_dir_0' in str(cell):
                label_loc = 0 #+ label_glob * 4
            elif '_dir_90' in str(cell):
                label_loc = 1 #+ label_glob * 4
            elif '_dir_180' in str(cell):
                label_loc = 2 #+ label_glob * 4
            elif '_dir_270' in str(cell):
                label_loc = 3 #+ label_glob * 4
            sequence.append(
                {'folder_path': cell,
                 'label_loc': label_loc,
                 'label_glob': label_glob,
                 'idx': i,
                }
                )
            
            i += 1
        
        self.fine_classification = fine_classification
        self.sequence = sequence
        self.nt = nt
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.transform = transform
        
    def __getitem__(self, index):
        
        v = self.sequence[index]
        vpath = v['folder_path']
        if self.fine_classification is True:
            target = v['label_loc']
        else:
            target = v['label_glob']
        
        # for every sample from the sequence, load all the frames
        seq = [pil_loader(os.path.join(vpath, 'zcmplx_rdk_%02d.png' % (i+1))) if v['label_glob'] == 1 else pil_loader(os.path.join(vpath, 'rdk_%02d.png' % (i+1))) for i in range(self.num_seq*self.seq_len)]
        t_seq = self.transform(seq)
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)
#         t_seq = t_seq.transpose(0,1)
        
        if self.return_label:
            return t_seq, target
        else:
            return t_seq
        
    def __len__(self):
        return len(self.sequence)
    


class CIFAR10_3d(CIFAR10):
    def __init__(self, root, train = True, transform = None, target_transform = None, download = False, seq_len=5, num_seq=8):
        super().__init__(root = root, 
                        train = train, 
                        transform = transform, 
                        target_transform = target_transform, 
                        download = download)
        self.seq_len = seq_len
        self.num_seq = num_seq

    def __getitem__(self, index):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (image, target) where target is index of the target class.
            """
            img, target = self.data[index], self.targets[index]
            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            seq = [Image.fromarray(img) for _ in range(self.seq_len)]
            
            if self.transform is not None:    
                t_seq = self.transform(seq)
            else:
                t_seq = seq
                
            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0)
            t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)            

            if self.target_transform is not None:
                target = self.target_transform(target)

            return t_seq, target


