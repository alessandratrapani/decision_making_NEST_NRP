import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from run_multiple_sim import *
import os
# FIGURE3: mostrare toin coss, simmetria rete
# sim 2 c0.0 una dove vince A una dove vince B, average 100x2 sim for decison space and delat S plot
fig_n = 'Figure3'
run = False
figure_3a = True
figure_3b = False
figure_3c = False
start_stim = 200.0
end_stim = 1200.0
simtime = 2500.0

if not os.path.exists('figures/'+fig_n+'/'):
    os.makedirs('figures/'+fig_n+'/')

if run:
    run_multiple_sim(mult_coherence=[0.0], start_stim = start_stim, end_stim = end_stim,simtime = simtime)

if figure_3a:
    save = True
    dt_string = 'standard/'
    
    fig = None
    ax_raster = None
    ax_rate = None
    fig, ((ax_raster_A,ax_raster_B), (ax_rate_A,ax_rate_B), (ax_stimuli_A,ax_stimuli_B), (ax_sum_stimuli_A,ax_sum_stimuli_B)) = plt.subplots(4, 2, sharex=True, figsize=(16,9))
    
    coherence = 0.0
    plt.suptitle('Coherence ' + str(coherence*100) + '%')

    win_pop = 'A_win'
    #DA SCEGLIERE
    j = 1
    path = 'results/'+dt_string+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j)+'/'
    evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(path, 'A')
    evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
    ax_raster_A.scatter(tsA, evsA,marker = '|', linewidths = 0.8, color='red', label ='pop A')
    ax_raster_A.scatter(tsB, evsB,marker = '|', linewidths = 0.8, color='blue', label ='pop B')
    ax_raster_A.vlines(start_stim, 0, 1600, color='grey')
    ax_raster_A.vlines(end_stim, 0, 1600, color='grey')
    ax_raster_A.set_ylabel("neuron #")
    ax_raster_A.set_title("Raster Plot ", fontsize=10)
    #ax_raster_A.legend()
    ax_rate_A.plot(t, A_N_A, color='red', label ='pop A')
    #ax_rate_A.fill_between(t, A_N_A, color='red')
    ax_rate_A.plot(t, B_N_B, color='blue', label ='pop B')
    #ax_rate_A.fill_between(t, B_N_B, color='blue')
    ax_rate_A.vlines(start_stim, 0, 40, color='grey')
    ax_rate_A.vlines(end_stim, 0, 40, color='grey')
    ax_rate_A.set_ylabel("A(t) [Hz]")
    ax_rate_A.set_title("Activity", fontsize=10)
    #ax_rate_A.legend()
    ax_stimuli_A.plot(np.arange(0., simtime),stimulus_A, 'red', label='stimulus on A')
    ax_stimuli_A.plot(np.arange(0., simtime),stimulus_B, 'blue', label='stimulus on B')
    ax_stimuli_A.set_ylabel("stimulus [Hz]")
    ax_stimuli_A.set_title("Stochastic inputs", fontsize=10)
    #ax_stimuli_A.legend()
    ax_sum_stimuli_A.plot(np.arange(0., simtime),sum_stimulus_A, 'red', label='sum_stimulus on A')
    ax_sum_stimuli_A.plot(np.arange(0., simtime),sum_stimulus_B, 'blue', label='sum_stimulus on B')
    ax_sum_stimuli_A.set_title("Time integral of inputs", fontsize=10)
    #ax_sum_stimuli_A.legend()
    plt.xlabel("t [ms]")

    win_pop = 'B_win'
    #DA SCEGLIERE
    j = 0
    path = 'results/'+dt_string+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j)+'/'
    evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(path, 'A')
    evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path,'B')
    ax_raster_B.scatter(tsA, evsA,marker = '|', linewidths = 0.8, color='red', label ='pop A')
    ax_raster_B.scatter(tsB, evsB,marker = '|', linewidths = 0.8, color='blue', label ='pop B')
    ax_raster_B.vlines(start_stim, 0, 1600, color='grey')
    ax_raster_B.vlines(end_stim, 0, 1600, color='grey')
    ax_raster_B.set_ylabel("neuron #")
    ax_raster_B.set_title("Raster Plot ", fontsize=10)
    #ax_raster_B.legend()
    ax_rate_B.plot(t, A_N_A, color='red', label ='pop A')
    #ax_rate_B.fill_between(t, A_N_A, color='red')
    ax_rate_B.plot(t, B_N_B, color='blue', label ='pop B')
    #ax_rate_B.fill_between(t, B_N_B, color='blue')
    ax_rate_B.vlines(start_stim, 0, 40, color='grey')
    ax_rate_B.vlines(end_stim, 0, 40, color='grey')
    ax_rate_B.set_ylabel("A(t) [Hz]")
    ax_rate_B.set_title("Activity", fontsize=10)
    #ax_rate_B.legend()
    ax_stimuli_B.plot(np.arange(0., simtime),stimulus_A, 'red', label='stimulus on A')
    ax_stimuli_B.plot(np.arange(0., simtime),stimulus_B, 'blue', label='stimulus on B')
    ax_stimuli_B.set_ylabel("stimulus [Hz]")
    ax_stimuli_B.set_title("Stochastic inputs", fontsize=10)    
    #ax_stimuli_B.legend()
    ax_sum_stimuli_B.plot(np.arange(0., simtime),sum_stimulus_A, 'red', label='sum_stimulus on A')
    ax_sum_stimuli_B.plot(np.arange(0., simtime),sum_stimulus_B, 'blue', label='sum_stimulus on B')
    ax_sum_stimuli_B.set_title("Time integral of inputs", fontsize=10)
    
    #ax_sum_stimuli_B.legend()
    plt.xlabel("t [ms]")
    if save:
        fig.savefig('figures/'+fig_n+'/Figure3A.png' , bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if figure_3b:
    fig, dec_space = plt.subplots(1, 1,  figsize=(3,3))
    dt_string = 'standard/'
    n_trial = 200
    coherence = 0.0
    win_pop = 'A_win'
    A_N_A_mean = []
    B_N_B_mean = []
    for j in range(n_trial): 
        path = 'results/'+dt_string+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j)+'/'
        if os.path.exists(path):
            evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(path, 'A')
            evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
            A_N_A_mean.append(A_N_A)
            B_N_B_mean.append(B_N_B)

    A_N_A_mean = np.mean(A_N_A_mean,axis=0)
    B_N_B_mean = np.mean(B_N_B_mean,axis=0)
    dec_space.plot(A_N_A,B_N_B, color='red', label ='pop A wins')

    win_pop = 'B_win'
    A_N_A_mean = []
    B_N_B_mean = []
    for j in range(n_trial): 
        path = 'results/'+dt_string+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j)+'/'
        if os.path.exists(path):
            evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(path, 'A')
            evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
            A_N_A_mean.append(A_N_A)
            B_N_B_mean.append(B_N_B)
    A_N_A_mean = np.mean(A_N_A_mean,axis=0)
    B_N_B_mean = np.mean(B_N_B_mean,axis=0)
    dec_space.plot(A_N_A,B_N_B, color='blue', label ='pop B wins')
    dec_space.plot([0,40],[0,40], color='grey')
    dec_space.set_xlim(-0.1,40)
    dec_space.set_ylim(-0.1,40)
    dec_space.set_xlabel("Firing rate pop A (Hz)")
    dec_space.set_ylabel("Firing rate pop B (Hz)")
    dec_space.set_title("Decision Space")
    dec_space.legend()
    if save:
        fig.savefig('figures/'+fig_n+'/Figure3B.png' , bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if figure_3c:
    save = True
    delta_s_A_winner = np.array([])
    delta_s_B_winner = np.array([])
    coherence = 0.0
    dt_string = 'standard/'
    results_dir= 'results/'+dt_string

    delta_s_A_winner=pd.read_csv(results_dir+'c'+str(coherence) +'/delta_s_A_winner.csv')
    delta_s_B_winner=pd.read_csv(results_dir+'c'+str(coherence) +'/delta_s_B_winner.csv')
    delta_s_A_winner=delta_s_A_winner['delta_s_A_winner'].to_numpy()
    delta_s_B_winner=delta_s_B_winner['delta_s_B_winner'].to_numpy()
    
    fig, ax1 = plt.subplots(1,1,figsize = [5,5])          
    ax1.hist(delta_s_A_winner, histtype = 'step', color = 'red', linewidth = 2)
    ax1.hist(delta_s_B_winner, histtype = 'step', color = 'blue', linewidth = 2)
    ax1.set_xlabel('Time integral of $s_1$(t) - $s_2$(t)')
    ax1.set_ylabel('Count #')

    if save:
        fig.savefig('figures/'+fig_n+'/Figure3C.png' , bbox_inches='tight')
        plt.close()
    else:
        plt.show()