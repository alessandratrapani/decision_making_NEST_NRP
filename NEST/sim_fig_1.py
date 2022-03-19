import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
text_color = 'black'
rcParams['text.color'] = text_color
rcParams['axes.labelcolor'] = text_color
rcParams['xtick.color'] = text_color
rcParams['ytick.color'] = text_color
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=26)  # fontsize of the figure title

import pandas as pd
import numpy as np
import scipy.signal as signal
import os
from run_multiple_sim import *
run_1 = False

#sim 1 c51.2 1 c12.8, 1 0.0, save also inhi activity, average 100 sim for decison space plot
order = 400
simtime = 3000.0
start_stim = 200.0
end_stim = 1200.0
dt = 0.1
dt_rec = 10.0

figure1abc = True
figure1ed = True
n_trial = 200
fig_n = 'Figure1'
if not os.path.exists('figures/'+fig_n+'/'):
    os.makedirs('figures/'+fig_n+'/')

if run_1:
    run_multiple_sim(n_trial = 200)

if figure1abc:
    fig1a,axes = plt.subplots(6, 2,  figsize=(6,10),sharex=True,constrained_layout=True)
    fig1de,ax = plt.subplots(2, 1,  figsize=(6,10),constrained_layout=True)
    mult_coherence = [0.0,-0.128,-0.512]
    color_A= ['tomato','r','darkred']
    color_inh = ['darkgrey','dimgrey','k']
    #DA SETTARE
    trial = [158,199,6]
    ax[1].plot([0,40],[0,40], color='grey')
    for i, coherence in enumerate(mult_coherence):
        j = trial[i]
        win_pop = 'A_win'
        results_dir = 'results/standard/'
        path = results_dir+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j)+'/'
        evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(path, 'A')
        evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
        evsIn, tsIn, t, I_N_I, stimulus_A, sum_stimulus_A = extract_results(path, 'inh')
        I_N_I=signal.medfilt(I_N_I,55) 
        ax[0].plot(t[:120], I_N_I[:120], color=color_inh[i],alpha= 0.9, label ='c '+str(coherence*100)+'%')
        ax[1].plot(A_N_A,B_N_B,color=color_A[i], alpha= 0.9, label ='c '+str(coherence*100)+'%')

        axes[2*i][0].scatter(tsA, evsA,marker = '|', linewidths = 0.8, color='red', label ='pop A')
        axes[2*i][0].set_ylabel("neuron #")
        axes[2*i][0].vlines(start_stim, 0, 800, color='grey')
        axes[2*i][0].vlines(end_stim, 0, 800, color='grey')
        axes[2*i][0].set_yticks([]) 
        axes[2*i+1][0].plot(t, A_N_A, color='red', label ='pop A')
        axes[2*i+1][0].fill_between(t, A_N_A, color='red')
        axes[2*i+1][0].vlines(start_stim, 0, 60, color='grey')
        axes[2*i+1][0].vlines(end_stim, 0, 60, color='grey')
        axes[2*i+1][0].set_ylabel("A(t) [Hz]")
        axes[2*i+1][0].set_ylim(0,60)

        axes[2*i][1].scatter(tsB, evsB,marker= '|',  linewidths = 0.8,color='blue', label ='pop B')
        axes[2*i][1].vlines(start_stim, 800, 1600, color='grey')
        axes[2*i][1].vlines(end_stim, 800, 1600, color='grey')
        axes[2*i][1].set_yticks([]) 
        axes[2*i+1][1].plot(t, B_N_B, color='blue', label ='pop B')
        axes[2*i+1][1].fill_between(t, B_N_B, color='blue')
        axes[2*i+1][1].vlines(start_stim, 0, 60, color='grey')
        axes[2*i+1][1].vlines(end_stim, 0, 60, color='grey')
        axes[2*i+1][1].set_ylim(0,60)
        axes[2*i+1][1].set_yticks([])

    axes[5][0].set_xlabel("t [ms]")
    axes[5][1].set_xlabel("t [ms]")

    ax[0].set_ylabel("A(t) [Hz]")
    ax[0].set_xlabel("t [ms]")
    ax[0].set_aspect(10)
    ax[0].set_ylim(0,60)
    ax[1].set_xlim(-0.1,40)
    ax[1].set_ylim(-0.1,40)
    ax[1].set_xlabel("Firing rate pop A (Hz)")
    ax[1].set_ylabel("Firing rate pop B (Hz)")    
    fig1a.savefig('figures/'+fig_n+'/Figure1a.png', bbox_inches='tight')
    fig1de.savefig('figures/'+fig_n+'/Figure1de.png', bbox_inches='tight')

if figure1ed:

    fig_1d, ax_rate_in = plt.subplots(1, 1,  figsize=(3,3))
    fig_1e, dec_space = plt.subplots(1, 1,  figsize=(3,3))
    mult_coherence = [0.0,-0.128,-0.512]
    
    win_pop = 'B_win'
    for i,coherence in enumerate(mult_coherence):
        A_N_A_mean = []
        B_N_B_mean = []
        I_N_I_mean = []
        for j in range(n_trial):  
            results_dir = 'results/standard/'
            path = results_dir+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j)+'/'
            if os.path.exists(path):
                evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(path, 'A')
                evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
                evsIn, tsIn, t, I_N_I, stimulus_A, sum_stimulus_A = extract_results(path, 'inh')
                
                A_N_A_mean.append(A_N_A)
                B_N_B_mean.append(B_N_B)
                I_N_I_mean.append(I_N_I)

        
        A_N_A_mean = np.mean(A_N_A_mean,axis=0)
        B_N_B_mean = np.mean(B_N_B_mean,axis=0)
        I_N_I_mean = np.mean(I_N_I_mean,axis=0)

        dec_space.plot(A_N_A_mean,B_N_B_mean, label ='c '+str(coherence*100)+'%')
        ax_rate_in.plot(I_N_I_mean,  label ='c '+str(coherence*100)+'%')

    ax_rate_in.set_ylabel("A(t) [Hz]")
    fig_1d.savefig('figures/'+fig_n+'/Figure1d_mean.png', bbox_inches='tight')
    
    dec_space.set_xlim(-0.1,40)
    dec_space.set_ylim(-0.1,40)
    dec_space.set_xlabel("Firing rate pop A (Hz)")
    dec_space.set_ylabel("Firing rate pop B (Hz)")
    fig_1e.savefig('figures/'+fig_n+'/Figure1e_mean.png', bbox_inches='tight')