import numpy as np
import torch
from tqdm import tqdm

class StimuliDataset(torch.utils.data.Dataset):
	def __init__(self,dataset,transform=None):
		'''
		dataset: numpy array of images/stimuli
		transform: transform to be applied to image
		'''
		self.images = dataset
		self.len_dataset = len(self.images)
		self.transform = transform

	def __len__(self):
		return self.len_dataset

	def __getitem__(self,index):
		img = self.images[index]
		if self.transform:
			img = self.transform(img)
		return img
		