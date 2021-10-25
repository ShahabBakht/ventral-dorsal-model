import os, sys
import allensdk
print(f'allen sdk version is {allensdk.__version__}')
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.brain_observatory_exceptions import NoEyeTrackingException
import allensdk.brain_observatory.dff as dff
import allensdk.brain_observatory.stimulus_info as stim_info

# from oasis_preprocessing import deconvolve_calcium_unknown
from CKA import *

import pprint
import numpy as np
import scipy as sp
from generate_SSM import *
import matplotlib.pyplot as plt
import pdb

boc = BrainObservatoryCache()
targeted_structures = boc.get_all_targeted_structures()
imaging_depths = boc.get_all_imaging_depths()

def get_RSM(CreLine = ['Cux2-CreERT2'], TargetedStruct = ['VISp'], ImagingDepth = [175], StimType = 'natural_scenes', downsample_rate = 1, pool_sessions = True):
    # get the number of exp containers (length of num_exps) and the number of experiments within each container (each element of num_exps)
    num_exps = get_number_experiments(CreLine = CreLine,TargetedStruct = TargetedStruct, ImagingDepth = ImagingDepth, StimType = StimType)
    num_exp_containers = len(num_exps)

    num_stim_conditions = get_num_stim_conditions(StimType = StimType)
    totalexp = 0
    if pool_sessions:
        num_iter = 100
        print(num_stim_conditions)
        all_RSM = np.ndarray((int(num_stim_conditions * downsample_rate),int(num_stim_conditions * downsample_rate),num_iter)) # multiplied by downsample_rate in the case the movie data is downsampled (only matters for the movie data)
        for eccounter in range(0,num_exp_containers):#
            for expcounter in range(0,num_exps[eccounter]):
                print(eccounter, expcounter)
                data_set, events = get_one_dataset(CreLine = CreLine, TargetedStruct = TargetedStruct, ImagingDepth = ImagingDepth, StimType = StimType, ExpContainerIdx = eccounter,ExpIdx = expcounter)

                if StimType == 'natural_scenes':
                    activations = get_activations_natuscene(data_set, events)

                elif StimType == 'drifting_gratings':
                    activations = get_activations_dg(data_set)

                elif StimType == 'static_gratings':
                    activations = get_activations_sg(data_set)

                elif (StimType == 'natural_movie_one')| (StimType == 'natural_movie_two') | (StimType == 'natural_movie_three'):
                    activations = get_activations_nm(data_set,StimType,downsample_rate, events)

                if totalexp == 0:
                    V1_tmp = activations['V1']
                else:
                    print('V1_tmp shape is ', V1_tmp.shape)
                    V1_tmp = np.concatenate((V1_tmp,activations['V1']),axis=2)

                del activations
                totalexp = totalexp + 1
                
        num_neurons = V1_tmp.shape[2]
        num_trials = V1_tmp.shape[0]
        print('total no. neurons =', num_neurons)
        
        noise_ceiling_rsm, noise_ceiling_cka = estimate_RSM_noise_ceiling(V1_tmp,num_trials)
        
        num_neurons_per_iter = num_neurons
        for itercount in range(0,num_iter):
            trials_idx_permute = np.random.permutation(np.arange(0,num_trials))
            which_trials = trials_idx_permute[0:int(num_trials/2)]

            this_V1 = np.mean(V1_tmp[which_trials,:,:],axis=0)
            activations = {'V1':this_V1}    
            activations_centered = center_activations(activations)
            sim_mats = compute_similarity_matrices(activations_centered)
            RSM_v1=sim_mats['V1']
            all_RSM[:,:,itercount] = RSM_v1
        
        if StimType == 'natural_movie_one': # estimate curvature of video sequence in V1 representation space
            temporal_size = this_V1.shape[0]
            num_iter = 10 # subsampling trials for multiple estimates
            c_v1 = np.zeros((num_iter,temporal_size))
            for j in range(num_iter):
                x = V1_tmp[np.random.permutation(V1_tmp.shape[0])[0:int(V1_tmp.shape[0]/2)],:,:]
                xt1 = np.mean(x,axis=0)
                xt2 = np.roll(xt1,axis = 0,shift=1)

                vt = xt1 - xt2
                v_norm = np.ndarray((vt.shape[0],1))
                v_norm[:,0] = np.linalg.norm(vt,axis=1)
                v_norm_tile = np.tile(v_norm,(1,vt.shape[1]))
                vt_normalized = vt/v_norm_tile
                for i in range(1,vt_normalized.shape[0]):
                    if i < vt_normalized.shape[0]-1:
                        v1 = vt_normalized[i,:]
                        v2 = vt_normalized[i+1,:]
                        c_v1[j,i] = (np.arccos(np.inner(v1,v2))*180/np.pi)
        
    elif ~pool_sessions:      
        all_RSM = np.ndarray((int(num_stim_conditions * downsample_rate),int(num_stim_conditions * downsample_rate),sum(num_exps))) # multiplied by downsample_rate in the case the movie data is downsampled (only matters for the movie data)
        for eccounter in range(0,num_exp_containers):
            for expcounter in range(0,num_exps[eccounter]):
                print(eccounter, expcounter)
                data_set, events = get_one_dataset(CreLine = CreLine, TargetedStruct = TargetedStruct, ImagingDepth = ImagingDepth, StimType = StimType, ExpContainerIdx = eccounter,ExpIdx = expcounter)

                if StimType == 'natural_scenes':
                    activations = get_activations_natuscene(data_set, event)

                elif StimType == 'drifting_gratings':
                    activations = get_activations_dg(data_set)

                elif StimType == 'static_gratings':
                    activations = get_activations_sg(data_set)

                elif (StimType == 'natural_movie_one')| (StimType == 'natural_movie_two') | (StimType == 'natural_movie_three'):
                    activations = get_activations_nm(data_set,StimType,downsample_rate, events)


                sim_mats = compute_similarity_matrices(activations)
                RSM_v1=sim_mats['V1']
                all_RSM[:,:,totalexp] = RSM_v1
                totalexp = totalexp + 1
        

    return all_RSM, noise_ceiling_rsm, noise_ceiling_cka, V1_tmp

def estimate_RSM_noise_ceiling(activations,num_trial):
    # activations is T (number of trials) x M (number of stimuli) x N (number of neurons)
    num_trials = activations.shape[0]
    print(activations[:,0,0])
    num_repetitions = num_trial
    r1 = np.empty([num_repetitions,1])
    r2 = np.empty([num_repetitions,1])
    for iter in range(0,num_repetitions):
        trials_idx_permute = np.random.permutation(np.arange(0,num_trials))
        which_trials = trials_idx_permute[0:int(num_trials/2)]
        other_trials = np.setdiff1d(np.arange(0,num_trials),which_trials)
        
        responses1 = activations[which_trials,:,:]
        responses2 = activations[other_trials,:,:]
        activation1 = {'V1':np.mean(responses1,axis=0)}  
        sim_mats1 = compute_similarity_matrices(activation1)
        activation2 = {'V1':np.mean(responses2,axis=0)}             
        sim_mats2 = compute_similarity_matrices(activation2)
        
        RSM1=sim_mats1['V1']
        RSM2=sim_mats2['V1']
        np.fill_diagonal(RSM1,'nan')
        np.fill_diagonal(RSM2,'nan')
        r1[iter] = compare_two_RSMs(RSM1,RSM2)
        
        r2[iter] = kernel_CKA(responses1.mean(0), responses2.mean(0)) 
    
    return r1, r2
        
def compare_two_RSMs(RSM1,RSM2):
    
    r=compute_ssm(RSM1, RSM2)
    return r

def compare_multi_RSMs(all_RSM):
    
    num_RSMs = all_RSM.shape[2]
    R = np.empty((num_RSMs,num_RSMs))
    for i in range(0,num_RSMs):
        for j in range(0,num_RSMs):
            R[i,j] = compare_two_RSMs(all_RSM[:,:,i],all_RSM[:,:,j])

            
    return R
    
    
def get_number_experiments(CreLine = ['Cux2-CreERT2'],TargetedStruct = ['VISp'],ImagingDepth = [175], StimType = 'natural_scenes'):
    all_ecs = boc.get_experiment_containers(cre_lines=CreLine,targeted_structures=TargetedStruct,imaging_depths=ImagingDepth)
    num_exp_containers = len(all_ecs)
    print("number of ", *CreLine, "experiment containers: %d\n" % num_exp_containers)
    num_exps = list()
    for eccounter in range(0,num_exp_containers):
        ec_id = all_ecs[eccounter]['id']
        exps = boc.get_ophys_experiments(experiment_container_ids=[ec_id], 
                                            stimuli=[StimType])
        
        num_exps.append(len(exps))
        print("experiment container: %d\n" % ec_id, ":", len(exps))
        
    return num_exps
    

def get_one_dataset(CreLine = ['Cux2-CreERT2'],TargetedStruct = ['VISp'],ImagingDepth = [175], StimType = 'natural_scenes', ExpContainerIdx = 0,ExpIdx = 0):
    
    all_ecs = boc.get_experiment_containers(cre_lines=CreLine,targeted_structures=TargetedStruct,imaging_depths=ImagingDepth)
    all_ec_id = all_ecs[ExpContainerIdx]['id']
    exp = boc.get_ophys_experiments(experiment_container_ids=[all_ec_id], 
                                        stimuli=[StimType])[ExpIdx]#
    data_set = boc.get_ophys_experiment_data(exp['id'])
    events = boc.get_ophys_experiment_events(exp['id'])
    
    return data_set, events

def get_num_stim_conditions(StimType = ['natural_scenes']):
    
    if StimType == 'static_gratings':
        all_orientations = [0,30,60,90,120,150]
        all_sf = [0.02,0.04,0.08,0.16,0.32]
        all_ph = [0,0.25,0.5,0.75]
        num_stim_conditions = len(all_orientations) * len(all_sf) * len(all_ph)
        return num_stim_conditions
    
    elif StimType == 'drifting_gratings':
        all_directions = [0,45,90,135,180,225,270,315]
        all_tf = [1,2,4,8,15]
        num_stim_conditions = len(all_directions) * len(all_tf)
        return num_stim_conditions
    
    elif StimType == 'natural_scenes':
        numImages = 118
        num_stim_conditions = numImages
        return num_stim_conditions
    
    elif StimType == 'natural_movie_one':
        num_stim_conditions = 900
        return num_stim_conditions
    
    elif StimType == 'natural_movie_two':
        num_stim_conditions = 900
        return num_stim_conditions
        
    elif StimType == 'natural_movie_three':
        num_stim_conditions = 3600
        return num_stim_conditions
    

def get_activations_sg(data_set):
    stim_table = data_set.get_stimulus_table('static_gratings')
    all_cell_ids = data_set.get_cell_specimen_ids()
    num_neurons = len(all_cell_ids)
    print('there are ' + str(num_neurons) + ' neurons in this session')
    all_orientations = [0,30,60,90,120,150]
    all_sf = [0.02,0.04,0.08,0.16,0.32]
    all_ph = [0,0.25,0.5,0.75]

    responses = np.empty([num_neurons,len(all_orientations)*len(all_sf)*len(all_ph)])

    for ncounter in range(0,num_neurons):
        _, sample_cell = data_set.get_dff_traces(cell_specimen_ids=[all_cell_ids[ncounter]])
        sample_cell = sample_cell[0]
        counter = 0
        for sfcount, sf in enumerate(all_sf):
            for orcount, ori in enumerate(all_orientations):
                for pcount, ph in enumerate(all_ph):

                    thisstim = stim_table[(stim_table.spatial_frequency == sf) & (stim_table.orientation == ori) & (stim_table.phase == ph)].to_numpy()
                    response_tmp = np.empty([1,thisstim.shape[0]])
                    for tr in range(0,thisstim.shape[0]):
                        response_tmp[0,tr] =  np.nanmean(sample_cell[int(thisstim[tr,3]):int(thisstim[tr,4])])

                    responses[ncounter,counter] = np.median(response_tmp)
                    counter = counter + 1


    print(responses.shape)  

    activations = {'V1':np.transpose(responses)}
    
    return activations
    
    
def get_activations_dg(data_set):
    stim_table = data_set.get_stimulus_table('drifting_gratings')
    all_cell_ids = data_set.get_cell_specimen_ids()
    num_neurons = len(all_cell_ids)
    print('there are ' + str(num_neurons) + ' neurons in this session')
    all_directions = [0,45,90,135,180,225,270,315]
    all_tf = [1,2,4,8,15]

    responses = np.empty([num_neurons,len(all_directions)*len(all_tf)])

    for ncounter in range(0,num_neurons):
        _, sample_cell = data_set.get_dff_traces(cell_specimen_ids=[all_cell_ids[ncounter]])
        sample_cell = sample_cell[0]
        counter = 0
        for tfcount, tf in enumerate(all_tf):
            for dircount, direct in enumerate(all_directions):

                thisstim = stim_table[(stim_table.temporal_frequency == tf) & (stim_table.orientation == direct)].to_numpy()
                response_tmp = np.empty([1,thisstim.shape[0]])
                for tr in range(0,thisstim.shape[0]):
                    response_tmp[0,tr] =  np.nanmean(sample_cell[int(thisstim[tr,3]):int(thisstim[tr,4])])

                responses[ncounter,counter] = np.median(response_tmp)
                counter = counter + 1


    print(responses.shape)  

    activations = {'V1':np.transpose(responses)}
    
    return activations

def get_activations_natuscene(data_set, events):
    
    stim_table = data_set.get_stimulus_table('natural_scenes')
    all_cell_ids = data_set.get_cell_specimen_ids()
    num_neurons = len(all_cell_ids)
    print('there are ' + str(num_neurons) + ' neurons in this session')
    numImages = 118
    numTrials = 50
    
    responses = np.empty([num_neurons,numImages,numTrials])

    for ncounter in range(0,num_neurons):

        sample_cell = events[ncounter,:]
        counter = 0
        for imcounter in range(0,numImages):

            thisstim = stim_table[(stim_table.frame == imcounter)].to_numpy()
            response_tmp = np.empty([1,thisstim.shape[0]])

            for tr in range(0,thisstim.shape[0]):
                response_tmp[0,tr] =  np.sum(sample_cell[int(thisstim[tr,1]):int(thisstim[tr,1])+15]) 
                
            
            responses[ncounter,counter,:] = response_tmp
            counter = counter + 1


    print(responses.shape)  

    activations_deconv = {'V1':np.transpose(responses)}

    return activations_deconv


def get_activations_nm(data_set,which_movie,downsample_rate, events):
    
    stim_table = data_set.get_stimulus_table(which_movie)
    try:
        timestamps, locations = data_set.get_pupil_location()

    except NoEyeTrackingException:
        print("No eye tracking for experiment %s." % data_set.get_metadata()["ophys_experiment_id"])
    
    all_cell_ids = data_set.get_cell_specimen_ids()
    num_neurons = len(all_cell_ids)
    print('there are ' + str(num_neurons) + ' neurons in this session')
    
    movie_len = len(stim_table[stim_table.repeat == 0])
    print('movie length is ' + str(movie_len))
    numTrials = 10
    responses = np.empty([num_neurons,int(movie_len * downsample_rate),numTrials])

    for ncounter in range(0,num_neurons):

        sample_cell = events[ncounter,:]
    
        response_tmp = np.empty([max(stim_table.repeat)+1,int(movie_len * downsample_rate)])
        for tr in range(0,max(stim_table.repeat)+1):
            start_time = stim_table.start[stim_table.repeat == tr] + 0#4
            this_cell_resp = sample_cell[start_time]
            if downsample_rate != 1:
                this_cell_resp_ds = downsample_movie_data(this_cell_resp,downsample_rate)
            else:
                this_cell_resp_ds = this_cell_resp
                
            response_tmp[tr,:] =  this_cell_resp_ds
        responses[ncounter,:,:] = np.transpose(response_tmp)


    print(responses.shape)  

    activations = {'V1':np.transpose(responses)}
    
    return activations

def get_eyes_nm(data_set,which_movie,downsample_rate):
    
    stim_table = data_set.get_stimulus_table(which_movie)
    movie_len = len(stim_table[stim_table.repeat == 0])
    print(movie_len)
    try:
        timestamps, locations = data_set.get_pupil_location()
        eye_avail = True

    except NoEyeTrackingException:
        eye_avail = False
        print("No eye tracking for experiment %s." % data_set.get_metadata()["ophys_experiment_id"])
    
    for tr in range(0,max(stim_table.repeat)+1):
        start_time = stim_table.start[stim_table.repeat == tr] + 4
        
        if eye_avail:

            which_eye_samples_start = ((timestamps) > start_time[tr*900]/30) 
            which_eye_samples_end = ((timestamps) < start_time[tr*900 + 899]/30)
            which_eye_samples = np.logical_and(which_eye_samples_start,which_eye_samples_end)
            
        else:
            pass
        

def downsample_movie_data(response,rate):
    num_samples = int(len(response) * rate)
    response_ds = sp.signal.resample(response,num_samples)
    
    return response_ds
    

if __name__ == '__main__':
    print('somethin')
