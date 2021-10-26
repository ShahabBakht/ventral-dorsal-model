import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from deepModelsAnalysis import *
from allenDataAnalysis import *
from CKA import *


def compare_reps(model_type = [('CPC','monkeynet','./epoch100.pth.tar')], StimType = 'natural_movies', area = [('VISp',275,'Cux2-CreERT2')], path_fig='./', compare_models = True, plot_hierarchy = False, tags = None):

    num_models = len(model_type)
    
    # get RSM for baseline models (now gabor3d, pixels)
    all_RSM_gabor, activations_gabor, model_gabor = get_GaborPyramid3D_RSMs(StimType = StimType,frame_per_block=5,ds_rate=3)
    
    all_RSM_pixel, activations_pixel = get_pixel_RSMs(StimType = StimType,frame_per_block=5,ds_rate=3)
    
    # get RSM for models
    num_areas = len(area)
    
    # mean and std of hierarchical level
    mean_level = []
    std_level = []
    
    for acounter in range(num_areas):
        
        this_area = area[acounter][0]
        this_depth = area[acounter][1]
        this_CreLine = area[acounter][2]
        print(f'\n ***** loading dataset area {this_area} _ {this_depth} _ {this_CreLine} ***** \n')

        all_RSM_Allen, noise_ceiling_rsm, noise_ceiling_cka, activations = get_RSM(CreLine = [this_CreLine], TargetedStruct = [this_area], 
                                            ImagingDepth = [this_depth], StimType = 'natural_movie_one',downsample_rate = 1/(3*5), 
                                            pool_sessions = True) 
        if 'all_noise_ceiling' not in locals():
            all_noise_ceiling = {f'{this_area}_{this_depth}_{this_CreLine}': noise_ceiling_rsm}
        else:
            all_noise_ceiling.update({f'{this_area}_{this_depth}_{this_CreLine}': noise_ceiling_rsm})
            
        print(f'\n ***** estimating similarity with baseline models ***** \n')
        # similarity with pixel RSM and gabor RSM
        num_sessions = all_RSM_Allen.shape[2]
        r_pixel = np.empty(num_sessions)
        r_pixel_cka = np.empty(num_sessions)
        r_gabor = np.empty(num_sessions)
        r_gabor_cka = np.empty(num_sessions)

        for j in range(0,num_sessions):
            r_pixel[j]=compute_ssm(all_RSM_Allen[:,:,j], all_RSM_pixel['pixel'])

            r_gabor[j]=compute_ssm(all_RSM_Allen[:,:,j], all_RSM_gabor['gabor'])
        
        if False:
            if 'RSM_similarity_max' not in locals():
                RSM_similarity_max = {(this_area+'_gabor'):r_gabor}
            else:
                RSM_similarity_max.update({(this_area+'_gabor'):r_gabor})



        for mcounter in range(num_models):
        
            # extract model info from the input args
            this_tag = tags[mcounter] if tags is not None else '_'
            this_model_type = model_type[mcounter][0] 
            this_backbone = model_type[mcounter][1]
            if len(model_type[mcounter]) == 2:
                pretrained = False
                this_path = ''
            elif len(model_type[mcounter]) == 3:
                pretrained = True
                this_path = model_type[mcounter][2]
            print(f'\n ***** estimating similarity with model {this_model_type} _ {this_backbone} _ pretrained_{pretrained} _ {this_tag} ***** \n')
            
            if this_model_type == 'CPC':
                all_RSM_model, activations_model, model = get_CPC_RSMs(StimType = StimType, backbone = this_backbone, pretrained =pretrained, path=this_path, frame_per_block = 5, ds_rate=3) 
            elif this_model_type == 'VGG':
                all_RSM_model, activations_model, model = get_othermodels_RSMs(StimType,'resnet18',frame_per_block=5,ds_rate=3)
            
            elif this_model_type == 'MonkeyNet':
                all_RSM_model, activations_model, model = get_monkeynet_RSMs(StimType,pretrained = pretrained, path = this_path,frame_per_block=5,ds_rate=3)
            
            elif this_model_type == 'ActionRecog':
                if this_backbone == 'monkeynet_p1':
                    all_RSM_model, activations_model, model = get_ActionRecog_RSMs(StimType,pretrained = pretrained, path = this_path,frame_per_block=5,ds_rate=3, num_paths = 1)
                elif this_backbone == 'monkeynet_p2':
                    all_RSM_model, activations_model, model = get_ActionRecog_RSMs(StimType,pretrained = pretrained, path = this_path,frame_per_block=5,ds_rate=3, num_paths = 2)
                
            # estimate similarity of model and mouse RSMs

            num_ignor_layers = 6
            if this_model_type == 'CPC':
                if this_backbone == 'mousenet' or this_backbone == 'simmousenet':
                    num_ignor_layers = 0
                elif this_backbone == 'vgg':
                    num_ignor_layers = 6
            elif this_model_type == 'VGG':
                num_ignor_layers = 0
            elif this_model_type == 'MonkeyNet':
                num_ignor_layers = 0
            elif this_model_type == 'ActionRecog':
                num_ignor_layers = 0

            r = np.empty([len(all_RSM_model.keys())-num_ignor_layers,num_sessions])
            for i in range(len(all_RSM_model.keys())-num_ignor_layers):
                act1 = activations_model[list(all_RSM_model.keys())[i]].mean(1).reshape(activations_model[list(all_RSM_model.keys())[i]].shape[0],-1)
                conv1 = all_RSM_model[list(all_RSM_model.keys())[i]]

                this_layer_name = list(all_RSM_model.keys())[i]


                print(list(all_RSM_model.keys())[i])
                if conv1.shape[0] >= all_RSM_Allen.shape[0]:
                    subsample_rate = int(conv1.shape[0]/all_RSM_Allen.shape[0])
                    conv1 = conv1[::subsample_rate,::subsample_rate]

                    for j in range(0,num_sessions):
                        act_tmp = activations.mean(0)
                        RSM_tmp = all_RSM_Allen[:,:,j]
                        r[i,j]=compute_ssm(conv1, RSM_tmp)
            
            # downsample layers ##
            r = downsample_r(r,backbone=this_backbone)
            ###
            r_mean = np.mean(r,axis=1)
            r_max_idx = np.argmax(r_mean)
            
            
            if 'RSM_similarity_max' not in locals():
                RSM_similarity_max = {(f'{this_area}_{this_model_type}_{this_backbone}_{str(pretrained)}_{this_tag}'):r[r_max_idx,:]}
            else:
                RSM_similarity_max.update({(f'{this_area}_{this_model_type}_{this_backbone}_{str(pretrained)}_{this_tag}'):r[r_max_idx,:]})

            # plots
            
            path_figure = path_fig + this_area + '/'
            thisFigureName = [f'{path_figure}{this_model_type}_{this_tag}_{str(pretrained)}_{StimType}_{str(this_depth)}_{this_CreLine}_ssm_wnoise.svg',
                              f'{path_figure}{this_model_type}_{this_tag}_{str(pretrained)}_{StimType}_{str(this_depth)}_{this_CreLine}_ssm.svg',
                              ]

            R = {'pixel': r_pixel,
                 'r': r
                 }

            noise_ceiling = {'rsm': noise_ceiling_rsm}
            
            # plot reps similarities per model/area
            plot_reps_sim(this_model_type, this_backbone, R, noise_ceiling, thisFigureName)
            
            # hierarchical level analysis
            m, s = hierarchical_level(R)
            mean_level.append(m)
            std_level.append(s)
            
            print(f'{this_area}: mean level = {m}, std level = {s}')
            
    if plot_hierarchy:
    # plot hierarchical levels    
        path_figure = path_fig + '/'
        thisFigureName = [f'{path_figure}{this_model_type}_{this_tag}_{str(pretrained)}_{StimType}_hierarchy.svg']
        plot_hierarchical_level(mean_level, std_level,thisFigureName) 
    
    if compare_models:        
    # plot model comaprisons
        plot_compare_models(RSM_similarity_max, all_noise_ceiling, [area[i][0] for i in range(len(area))], path_fig)


def hierarchical_level(R):
    
    r = R['r']
    print(r.shape)
    idx = np.argmax(r,0)
    for i in range(len(idx)):
        if idx[i] > 32:
            idx[i] -= 31
    m = idx.mean()
    s = idx.std()
    
    return m/32,s/32
    
def plot_hierarchical_level(mean_level, std_level,figure_name):
    print(f'mean: {mean_level}\n std:{std_level}')
    fig,ax = plt.subplots(figsize=(3,5))
    plt.scatter(np.arange(0,len(mean_level),1),mean_level,s = 100,c='gray',alpha=.8)
    plt.errorbar(np.arange(0,len(mean_level),1),mean_level, xerr=0, yerr=[i/2 for i in std_level],color='black',fmt='.')
    plt.grid()
    plt.xticks(np.arange(0,len(mean_level),1),['VISp','VISlm','VISpm','VISal','VISam']) #'VISal',,'VISam'['VISal','VISam']
    plt.xlabel('Areas',fontsize = 15, labelpad = 20)
    plt.ylabel('Hierarchy index',fontsize = 15, labelpad = 10)
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.savefig(figure_name[0], dpi = 300,bbox_inches='tight')
    plt.show()
    
def plot_reps_sim(model_type, backbone, R, noise_ceiling, FigurePath):
    
    r_pixel = R['pixel']
    r = R['r']
    noise_ceiling_rsm = noise_ceiling['rsm']
    
    # set layers hierarchical level
    color_map = np.array(['black','blue','red','green','orange','cyan','purple','brown','black','blue','red'])
    if model_type == 'CPC':
        if backbone == 'mousenet':
            hierarchical_levels = np.array([1,2,3,4,5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,8,9,10])
            areas_label = np.array([0,1,1,1,2,3,4,5,6,2,3,4,5,6,2,3,4,5,6,7,7,7])

        elif backbone == 'vgg':
            hierarchical_levels = np.array(range(1,14))
            areas_label = np.array([0])

        elif backbone == 'monkeynet_p2':

            hierarchical_levels = np.array(np.concatenate(([1],range(2,12),range(2,12))), dtype = np.int) 
            areas_label = np.array(np.concatenate((np.zeros((1)),np.ones((10)),2* np.ones((10)))), dtype = np.int)
            
        elif backbone == 'monkeynet_p1':

            hierarchical_levels = np.array(np.concatenate(([1],range(2,12))), dtype = np.int) 
            areas_label = np.array(np.concatenate((np.zeros((1)),np.ones((10)))), dtype = np.int)
            
        elif backbone == 'simmousenet':
            hierarchical_levels = np.array([1,2,3,4,4,5,6,6,5,6,6,5,6,6,5,6,6]) 
            areas_label = np.array([0,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6])
            
    elif model_type == 'MonkeyNet':
        if backbone == 'monkeynet_p2' or backbone == 'monkeynet_p1':
            hierarchical_levels = np.array(range(1,16))
            areas_label = np.array([0])
    
    elif model_type == 'VGG':
        if backbone == 'resnet18':
            hierarchical_levels = np.array(range(1,21))
            areas_label = np.array([0])

    elif model_type == 'ActionRecog':
        if backbone == 'monkeynet_p2':

            hierarchical_levels = np.array(np.concatenate(([1],range(2,12),range(2,12))), dtype = np.int) 
            areas_label = np.array(np.concatenate((np.zeros((1)),np.ones((10)),2* np.ones((10)))), dtype = np.int)
            
        elif backbone == 'monkeynet_p1':

            hierarchical_levels = np.array(np.concatenate(([1],range(2,12))), dtype = np.int) 
            areas_label = np.array(np.concatenate((np.zeros((1)),np.ones((10)))), dtype = np.int)
            
    print(r)    
    # create plots    
    
    
    csfont = {'size'   : 13}
    matplotlib.rc('font', **csfont)
    
    plt.scatter(0,np.median(r_pixel)/np.median(noise_ceiling_rsm),s = 100,c='gray',alpha=.8)
    plt.errorbar(0,np.median(r_pixel)/np.median(noise_ceiling_rsm),xerr=0, yerr=np.std(r_pixel/np.median(noise_ceiling_rsm))/2,color='black',fmt='.')

    plt.errorbar(hierarchical_levels,np.median(r,1)/np.median(noise_ceiling_rsm),xerr=0, yerr=np.std(r/np.median(noise_ceiling_rsm),1)/2,color='black',fmt='.')
    plt.scatter(hierarchical_levels,np.median(r,1)/np.median(noise_ceiling_rsm),s = 100,color=color_map[areas_label],alpha=.8)
    plt.grid()
    plt.ylim((0.3,0.75))
    plt.xlabel('Layers',fontsize = 15, labelpad = 20)
    plt.ylabel('RSM similarity\n(noise corrected)',fontsize = 15, labelpad = 10)
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False
                    ) # labels along the bottom edge are off
    plt.savefig(FigurePath[1],dpi=300,bbox_inches='tight')
    plt.show()
    
    plt.scatter(0,np.median(r_pixel),s = 100,c='gray',alpha=.8)
    plt.errorbar(0,np.median(r_pixel),xerr=0, yerr=np.std(r_pixel)/2,color='black',fmt='.')
    plt.fill_between(np.arange(0,np.max(hierarchical_levels)+1), np.transpose(np.median(noise_ceiling_rsm)-np.std(noise_ceiling_rsm)/2), 
                     np.transpose(np.median(noise_ceiling_rsm)+np.std(noise_ceiling_rsm)/2),color='green',alpha=0.5)
    plt.errorbar(hierarchical_levels,np.median(r,1),xerr=0, yerr=np.std(r,1)/2,color='black',fmt='.')
    plt.scatter(hierarchical_levels,np.median(r,1),s = 100,color=color_map[areas_label],alpha=.8)
    plt.grid()
    plt.xlabel('Layers',fontsize = 15, labelpad = 20)
    plt.ylabel('RSM similarity',fontsize = 15, labelpad = 10)
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False
                    ) # labels along the bottom edge are off
    plt.savefig(FigurePath[0],dpi=300,bbox_inches='tight')
    plt.show()

    
def downsample_r(r,backbone = 'monkeynet_p1'):
    
    # only for monkeynet architectures
    
    num_layers, num_sessions = r.shape
    if backbone == 'monkeynet_p1':
        r_ds = np.empty((11,num_sessions))
    elif backbone == 'monkeynet_p2':
        r_ds = np.empty((21,num_sessions))
        
    l = 0
    for l in range(r_ds.shape[0]):
        if l == 0:
            r_ds[l,:] = r[0:2,:].mean(0)
        elif l == 1:
            r_ds[l,:] = r[2:6,:].mean(0)
        else:
            if l <= 11:
                start_l = 2 + 4 + (l-2)*3
            else:
                start_l = 2 + 4 + 1 + (l-2)*3
            if l == 11:
                end_l= start_l + 4
            else:
                end_l= start_l + 3
            r_ds[l,:] = r[start_l:end_l,:].mean(0)
            
            
    return r_ds

    
    
def plot_compare_models(RSM_similarity_max, all_noise_ceiling, area, path_fig):
    
    print(RSM_similarity_max.keys())
    noise_c = []
    for k,a in all_noise_ceiling.items():
        noise_c.append(a.mean())
        print(f'noise ceiling for {k} = {a.mean()}')
    
    epochs_to_plot = [0.1,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, len(noise_c)))
    area_counter = 0
    for a in area:
        model_idx = 0
        ticks = []
        fig,ax = plt.subplots(figsize=(3.9,5))
        for key, value in RSM_similarity_max.items():
            if (a+'_') in key:
                ticks.append(key)
                fig_name = path_fig + a + '.svg' 
                plt.scatter(model_idx, np.mean(value/noise_c[area_counter]), marker='.',s=600, color='k') 
                plt.errorbar(model_idx, np.mean(value/noise_c[area_counter]), xerr = 0, yerr = np.std(value/noise_c[area_counter])/2, color='k') 
                model_idx += 1
        area_counter += 1
        
        plt.xticks(np.arange(model_idx), ticks,rotation='vertical') 
        ax.tick_params(labelsize=15)
        plt.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=True
                    ) # labels along the bottom edge are off
        plt.yticks()
        plt.grid(axis='both', which='both')
        plt.ylabel('Max RSM similarity\n(noise corrected)',fontsize = 15, labelpad = 20)
        plt.savefig(fig_name, dpi=300,bbox_inches='tight')
        plt.show()
        