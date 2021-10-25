import numpy as np
import torch
import sklearn
import sklearn.decomposition
import sklearn.random_projection
from tqdm import tqdm
from Stimuli import StimuliDataset
import sys, os
import pdb
import time



def compute_corr(Yl, Yp):
    corr = torch.zeros(Yl.shape[1], device=Yl.device)
    for i in range(Yl.shape[1]):
        yl, yp = (Yl[:, i].cpu().detach().numpy(), 
                  Yp[:, i].cpu().detach().numpy())
        yl = yl[~np.isnan(yl)]
        yp = yp[~np.isnan(yp)]
        corr[i] = np.corrcoef(yl, yp)[0, 1]
    return corr

def get_projection_matrix(X, n):
    X_ = X.cpu().detach().numpy()
    svd = sklearn.decomposition.TruncatedSVD(n_components=n, random_state=0xadded)
    r = svd.fit_transform(X_)
    return torch.tensor(svd.components_.T / r[:, 0].std(), device=X.device)

def compute_similarity_matrices(feature_dict, layers=None):
	'''
	feature_dict: a dictionary containing layer activations as numpy arrays
	layers: list of model layers for which features are to be generated

	Output: a dictionary containing layer activation similarity matrices as numpy arrays
	'''
	similarity_mat_dict = {}
	if layers is not None:
		for layer in layers:
			try:
				activation_arr = feature_dict[layer]
				activations_flattened = activation_arr.reshape((activation_arr.shape[0],-1))
				similarity_mat_dict[layer] = sim_pearson(activations_flattened) #np.corrcoef(activations_flattened)
			except Exception as e:
				print(layer)
				raise e
	else:
		for layer,activation_arr in feature_dict.items():
			try:
				activations_flattened = activation_arr.reshape((activation_arr.shape[0],-1))
				similarity_mat_dict[layer] = sim_pearson(activations_flattened) #np.corrcoef(activations_flattened)
			except Exception as e:
				print(layer,activation_arr.shape)
				raise e

	return similarity_mat_dict

def shuffle_similarity_mat(similarity_mat):
	'''
	similarity_mat: similarity matrix as a numpy array of size n X n

	Output: a random permuted order of similarity_mat (rows and columns permuted using the same order, i.e. order of stimuli changed)
	'''
	n = similarity_mat.shape[0]
	p = np.random.permutation(n)
	random_similarity_mat = similarity_mat[p]	# permute the rows
	random_similarity_mat = (random_similarity_mat.T[p]).T 	# permute the columns
	return random_similarity_mat

def sim_pearson(X):
    # X is [dim, samples]
    dX = (X.T - np.mean(X.T, axis=0)).T
    sigma = np.sqrt(np.mean(dX**2, axis=1)) + 1e-7

    cor = np.dot(dX, dX.T)/(dX.shape[1]*sigma)
    cor = (cor.T/sigma).T

    return cor

def compute_traj_smm(similarity1, similarity2):
    
    try:
        from scipy.stats import spearmanr
        from scipy.stats import kendalltau
        
        lowerdiag1 = np.diag(similarity1, k = -1)
        lowerdiag2 = np.diag(similarity2, k = -1)
        
        r,_ = kendalltau(lowerdiag1,lowerdiag2)
        return r
    except:
        print("Error in calculating spearman correlation")
        raise

    
def compute_ssm(similarity1, similarity2, num_shuffles=None, num_folds=None):
    '''
	similarity1: first similarity matrix as a numpy array of size n X n
	similarity2: second similarity matrix as a numpy array of size n X n
	num_shuffles: Number of shuffles to perform to generate a distribution of SSM values
	num_folds: Number of folds to split stimuli set into
    
	Output: the spearman rank correlation of the similarity matrices
    '''
    if num_shuffles is not None:
        raise NotImplementedError()
        
    if num_folds is not None:
	    raise NotImplementedError()
    
    try:
        from scipy.stats import spearmanr
        from scipy.stats import kendalltau
        lowertri_idx = np.tril_indices(similarity1.shape[0],k=-1)
        similarity1_lowertri = similarity1[lowertri_idx]
        similarity2_lowertri = similarity2[lowertri_idx]
#         r,_ = spearmanr(similarity1_lowertri,similarity2_lowertri)#,
        r,_ = kendalltau(similarity1_lowertri,similarity2_lowertri)
#         r,_ = spearmanr(similarity1.flatten(),similarity2.flatten())#,,nan_policy='omit'
        return r
    except:
	    print("Error in calculating spearman correlation")
	    raise

        
def center_activations(feature_dict):
    feature_dict_centered = dict()
    for layers, activation_arr in feature_dict.items():
        activation_flat = activation_arr.reshape((activation_arr.shape[0],-1))
#         pdb.set_trace()
        if torch.is_tensor(activation_flat):
            activation_flat = activation_flat.numpy()
            
        activation_mean_percolumn = np.mean(activation_flat,axis=0)
        activation_mean = np.tile(activation_mean_percolumn,(activation_flat.shape[0],1))
        activation_centered = activation_flat - activation_mean
        activation_centered_unflat = activation_centered.reshape((activation_arr.shape))
        feature_dict_centered[layers] = activation_centered_unflat
        
    return feature_dict_centered
        
        
    
    
if __name__=="__main__":
	# A placeholder function to illustrate usage of the functions
	N_stim = 100
	A = np.random.rand(N_stim,300)	# random array containing activations of 300 units
	B = np.random.rand(N_stim,500)	# random array containing activations of 500 units
	A_corrupt = 2*A + 0.01*np.random.rand(*A.shape)	# some corrupt version of 2*A
	activation_dict = {'L1':A,'L1_corrupt':A_corrupt,'L2':B}
	similarity_dict = compute_similarity_matrices(activation_dict)
	ssm_12 = compute_ssm(similarity_dict['L1'],similarity_dict['L2'])
	ssm_12_corrupt = compute_ssm(similarity_dict['L1_corrupt'],similarity_dict['L2'])
	ssm_11_corrupt = compute_ssm(similarity_dict['L1'],similarity_dict['L1_corrupt'])
	print("SSM values for 2 layer representations :",ssm_12)
	print("SSM values for 2 layer representations with noise :",ssm_12_corrupt)
	print("SSM values for a layer representations (with and without noise):",ssm_11_corrupt)