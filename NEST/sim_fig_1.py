import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
figure1ed = False

fig_n = 'Figure1'
if not os.path.exists('figures/'+fig_n+'/'):
    os.makedirs('figures/'+fig_n+'/')

if run_1:
    run_multiple_sim(n_trial = 200)

if figure1abc:
    fig1a,axes = plt.subplots(6, 2,  figsize=(5,10))
    fig1d,ax = plt.subplots(1, 1,  figsize=(3,3))
    fig1e,ax1e = plt.subplots(1, 1,  figsize=(3,3))
    mult_coherence = [0.0,-0.128,-0.512]
    #DA SETTARE
    trial = [0,0,1]
    for i, coherence in enumerate(mult_coherence):
        j = trial[i]
        win_pop = 'A_win'
        results_dir = 'results/standard/'
        path = results_dir+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j)+'/'
        evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(path, 'A')
        evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
        evsIn, tsIn, t, I_N_I, stimulus_A, sum_stimulus_A = extract_results(path, 'inh')
        
        ax.plot(t[:120], I_N_I[:120], alpha= 0.9, label ='c '+str(coherence*100)+'%')
        ax1e.plot(A_N_A,B_N_B,color=[0.4+i*0.2,0,0], alpha= 0.7, label ='c '+str(coherence*100)+'%')

        axes[2*i][0].scatter(tsA, evsA,marker = '|', linewidths = 0.8, color='red', label ='pop A')
        axes[2*i][0].set_ylabel("neuron #")
        axes[2*i][0].vlines(start_stim, 0, 800, color='grey')
        axes[2*i][0].vlines(end_stim, 0, 800, color='grey')
        axes[2*i][0].set_title("Raster Plot pop A ", fontsize=10)
        axes[2*i+1][0].plot(t, A_N_A, color='red', label ='pop A')
        axes[2*i+1][0].fill_between(t, A_N_A, color='red')
        axes[2*i+1][0].vlines(start_stim, 0, 40, color='grey')
        axes[2*i+1][0].vlines(end_stim, 0, 40, color='grey')
        axes[2*i+1][0].set_ylabel("A(t) [Hz]")
        axes[2*i+1][0].set_title("Activity Pop A", fontsize=10)

        axes[2*i][1].scatter(tsB, evsB,marker= '|',  linewidths = 0.8,color='blue', label ='pop B')
        axes[2*i][1].set_ylabel("neuron #")
        axes[2*i][1].set_title("Raster Plot pop B ", fontsize=10)
        axes[2*i][1].vlines(start_stim, 800, 1600, color='grey')
        axes[2*i][1].vlines(end_stim, 800, 1600, color='grey')
        axes[2*i+1][1].plot(t, B_N_B, color='blue', label ='pop B')
        axes[2*i+1][1].fill_between(t, B_N_B, color='blue')
        axes[2*i+1][1].vlines(start_stim, 0, 40, color='grey')
        axes[2*i+1][1].vlines(end_stim, 0, 40, color='grey')
        axes[2*i+1][1].set_ylabel("A(t) [Hz]")
        axes[2*i+1][1].set_title("Activity Pop B", fontsize=10)
        plt.xlabel("t [ms]")

    ax.set_ylabel("A(t) [Hz]")
    ax.set_title("Activity Pop Inh", fontsize=10)
    ax.legend()

    ax1e.set_xlim(-0.1,40)
    ax1e.set_ylim(-0.1,40)
    ax1e.set_xlabel("Firing rate pop A (Hz)")
    ax1e.set_ylabel("Firing rate pop B (Hz)")
    ax1e.set_title("Decision Space")
    ax1e.legend()
    fig1e.savefig('figures/'+fig_n+'/Figure1e.png', bbox_inches='tight')

    fig1a.savefig('figures/'+fig_n+'/Figure1a.png', bbox_inches='tight')
    fig1d.savefig('figures/'+fig_n+'/Figure1d.png', bbox_inches='tight')

if figure1ed:

    fig_1d, ax_rate_in = plt.subplots(1, 1,  figsize=(3,3))
    fig_1e, dec_space = plt.subplots(1, 1,  figsize=(3,3))
    mult_coherence = [0.0, -0.032, 0.032]
    n_trial = 200
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
    ax_rate_in.set_title("Activity Pop Inh", fontsize=10)
    ax_rate_in.legend()
    fig_1d.savefig('figures/'+fig_n+'/Figure1d_mean.png', bbox_inches='tight')
    
    dec_space.set_xlim(-0.1,40)
    dec_space.set_ylim(-0.1,40)
    dec_space.set_xlabel("Firing rate pop A (Hz)")
    dec_space.set_ylabel("Firing rate pop B (Hz)")
    dec_space.set_title("Decision Space")
    dec_space.legend()
    fig_1e.savefig('figures/'+fig_n+'/Figure1e_mean.png', bbox_inches='tight')