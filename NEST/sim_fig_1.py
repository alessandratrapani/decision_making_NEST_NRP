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

save = False

figure1abc = True
figure1ed = True

fig_n = 'Figure1'
if not os.path.exists('results/'+fig_n+'/'):
    os.makedirs('results/'+fig_n+'/')


if run_1:
    run_multiple_sim(n_trial = 200)

if figure1abc:
    fig1a,axes = plt.subplots(6, 2,  figsize=(5,10))
    fig1d,ax = plt.subplots(1, 1,  figsize=(3,3))
    mult_coherence = [0.0,0.128,0.512]
    for i, coherence in enumerate(mult_coherence):
        j = 0
        win_pop = 'win_B'
        results_dir = 'results/standard/'
        path = results_dir+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j)
        notes, evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(path, 'A')
        notes, evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
        notes, evsA, tsA, t, I_N_I, stimulus_A, sum_stimulus_A = extract_results(path, 'inh')
        
        ax.plot(t, I_N_I, color='black',alpha=1-i*0.1, label ='coh_' + '0-'+ str(coherence)[2:])

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

    fig1a.savefig('results/'+fig_n+'/Figure1a.eps', bbox_inches='tight')
    fig1d.savefig('results/'+fig_n+'/Figure1d.eps', bbox_inches='tight')

if figure1ed:

    fig_1d, ax_rate_in = plt.subplots(1, 1,  figsize=(3,3))
    fig_1e, dec_space = plt.subplots(1, 1,  figsize=(3,3))
    mult_coherence = [0.0, 0.128, 0.512]
    n_trial = 200
    win_pop = 'win_B'
    for i,coherence in enumerate(mult_coherence):
        A_N_A_mean = np.array([])
        B_N_B_mean = np.array([])
        I_N_I_mean = np.array([])
        for j in range(n_trial):  
            results_dir = 'results/standard/'
            path = results_dir+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j)
            if os.path.exists(path):
                activity = pd.read_csv(path)
                A_N_A = activity['activity (Hz) pop_A'].to_numpy()
                A_N_A_mean = np.append(A_N_A_mean,A_N_A,axis=0)
                B_N_B = activity['activity (Hz) pop_B'].to_numpy()
                B_N_B_mean = np.append(B_N_B_mean,B_N_B,axis=0)
                I_N_I = activity['activity (Hz) pop_inh'].to_numpy()
                I_N_I_mean = np.append(I_N_I_mean,I_N_I,axis=0)

        A_N_A_mean = np.mean(A_N_A_mean)
        B_N_B_mean = np.mean(B_N_B_mean)
        I_N_I_mean = np.mean(I_N_I_mean)

        dec_space.plot(A_N_A_mean,B_N_B_mean, color='blue', alpha=1-i*0.1, label ='coh_' + '0-'+ str(coherence)[2:])
        ax_rate_in.plot(t,I_N_I_mean, color='black', alpha=1-i*0.1, label ='coh_' + '0-'+ str(coherence)[2:])

    ax_rate_in.set_ylabel("A(t) [Hz]")
    ax_rate_in.set_title("Activity Pop Inh", fontsize=10)
    ax_rate_in.legend()
    fig1a.savefig('results/'+fig_n+'/Figure1d_mean.eps', bbox_inches='tight')
    
    dec_space.set_xlim(-0.1,40)
    dec_space.set_ylim(-0.1,40)
    dec_space.set_xlabel("Firing rate pop A (Hz)")
    dec_space.set_ylabel("Firing rate pop B (Hz)")
    dec_space.set_title("Decision Space")
    dec_space.legend()
    fig_1e.savefig('results/'+fig_n+'/Figure1e.eps', bbox_inches='tight')