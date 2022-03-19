import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from run_multiple_sim import *
import os
# FIGURE2: mostrare effetto dei pesi excitatory recurrent
# sim 1 c51.2 decrease, 1 c0.0 decrease, 100 6.4 increase, 1 c51.2 noNMDA
fig_n = 'Figure2'

run_1 = False
run_2 = False

figure_2a = False
figure_2b = False
figure_2c = True

order = 400
simtime = 3000.0
start_stim = 200.0
end_stim = 1200.0
dt = 0.1
dt_rec = 10.0
n_trial=100

if not os.path.exists('figures/'+fig_n+'/'):
    os.makedirs('figures/'+fig_n+'/')

if run_1:
    dt_string='decrease_w'
    results_dir = 'results/'+dt_string+'/'
    run_multiple_sim(results_dir =results_dir,n_trial =n_trial,mult_coherence = [0.0, 0.512],fn_tuned_par = "w_decrease.csv")

if figure_2a:
    fig_2a, axes = plt.subplots(2, 2, sharey=True,sharex=True,  figsize=(10,10))
    win_pop ='B_win'
    mult_coherence = [0.0, 0.512]

    for i, coherence in enumerate(mult_coherence):
        B_N_B_mean_decrease =[]
        B_N_B_mean =[]
        A_N_A_mean_decrease =[]
        A_N_A_mean =[]

        for j in range(n_trial):
            dt_string='decrease_w'
            path_decrease = 'results/'+dt_string+'/c'+str(coherence) +'/trial_'+ str(j) + '/'
            if os.path.exists(path_decrease):
                evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path_decrease, 'B')
                B_N_B_mean_decrease.append(B_N_B)
                evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(path_decrease, 'A')
                A_N_A_mean_decrease.append(A_N_A)

            dt_string='standard'
            path_decrease = 'results/'+dt_string+'/c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j) + '/'
            if os.path.exists(path_decrease):
                evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path_decrease, 'B')
                B_N_B_mean.append(B_N_B)
                evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(path_decrease, 'A')
                A_N_A_mean.append(A_N_A)

        B_N_B_mean_decrease = np.mean(B_N_B_mean_decrease,axis=0)
        B_N_B_smooth_decrease=signal.medfilt(B_N_B_mean_decrease,55) 
        B_N_B_mean = np.mean(B_N_B_mean,axis=0)
        B_N_B_smooth=signal.medfilt(B_N_B_mean,55) 
        A_N_A_mean_decrease = np.mean(A_N_A_mean_decrease,axis=0)
        A_N_A_smooth_decrease=signal.medfilt(A_N_A_mean_decrease,55) 
        A_N_A_mean = np.mean(A_N_A_mean,axis=0)
        A_N_A_smooth=signal.medfilt(A_N_A_mean,55) 

        axes[i][0].plot(t,B_N_B_mean_decrease,'blue')
        axes[i][0].plot(t,B_N_B_smooth_decrease,'green')
        axes[i][0].plot(t,B_N_B_mean,'blue',alpha=0.6)
        axes[i][0].plot(t,B_N_B_smooth,'green')
        axes[i][0].set_ylim(0,60)
        axes[i][0].set_ylabel("A(t) [Hz]")
        axes[i][0].vlines(start_stim, 0, 60, color='grey')
        axes[i][0].vlines(end_stim, 0, 60, color='grey')
        axes[i][1].plot(t,A_N_A_mean_decrease,'red')
        axes[i][1].plot(t,A_N_A_smooth_decrease,'green')
        axes[i][1].plot(t,A_N_A_mean,'red',alpha=0.6)
        axes[i][1].plot(t,A_N_A_smooth,'green')
        axes[i][1].set_ylim(0,60)
        axes[i][1].vlines(start_stim, 0, 60, color='grey')
        axes[i][1].vlines(end_stim, 0, 60, color='grey')
        
    axes[1][0].set_xlabel("t [ms]")
    axes[1][1].set_xlabel("t [ms]")
    fig_2a.savefig('figures/'+fig_n+'/'+'Figure2a.png' , bbox_inches='tight')
    plt.close()

if figure_2b: 
    coherence = 0.064
    
if run_2:
    dt_string='no_NMDA'
    results_dir = 'results/'+dt_string+'/'
    run_multiple_sim(results_dir =results_dir,n_trial =n_trial,mult_coherence = [0.0, 0.512],fn_fixed_par= "no_NMDA.csv")

if figure_2c:
    fig_2c, axes = plt.subplots(2, 2, sharey=True, sharex='col',  figsize=(10,10))
    win_pop ='B_win'
    mult_coherence = [0.0, 0.512]

    for i, coherence in enumerate(mult_coherence):
        B_N_B_mean_decrease =[]
        B_N_B_mean =[]
        A_N_A_mean_decrease =[]
        A_N_A_mean =[]

        for j in range(n_trial):
            dt_string='no_NMDA'
            path_decrease = 'results/'+dt_string+'/c'+str(coherence) +'/trial_'+ str(j) + '/'
            if os.path.exists(path_decrease):
                evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path_decrease, 'B')
                B_N_B_mean_decrease.append(B_N_B)
                evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(path_decrease, 'A')
                A_N_A_mean_decrease.append(A_N_A)

            dt_string='standard'
            path_decrease = 'results/'+dt_string+'/c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j) + '/'
            if os.path.exists(path_decrease):
                evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path_decrease, 'B')
                B_N_B_mean.append(B_N_B)
                evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(path_decrease, 'A')
                A_N_A_mean.append(A_N_A)

        B_N_B_mean_decrease = np.mean(B_N_B_mean_decrease,axis=0)
        B_N_B_smooth_decrease=signal.medfilt(B_N_B_mean_decrease,55) 
        B_N_B_mean = np.mean(B_N_B_mean,axis=0)
        B_N_B_smooth=signal.medfilt(B_N_B_mean,55) 
        A_N_A_mean_decrease = np.mean(A_N_A_mean_decrease,axis=0)
        A_N_A_smooth_decrease=signal.medfilt(A_N_A_mean_decrease,55) 
        A_N_A_mean = np.mean(A_N_A_mean,axis=0)
        A_N_A_smooth=signal.medfilt(A_N_A_mean,55) 

        axes[i][0].plot(t,B_N_B_mean_decrease,'blue')
        axes[i][0].plot(t,B_N_B_smooth_decrease,'green')
        axes[i][0].plot(t,B_N_B_mean,'blue',alpha=0.6)
        axes[i][0].plot(t,B_N_B_smooth,'green')
        axes[i][0].set_ylim(0,60)
        axes[i][0].set_ylabel("A(t) [Hz]")
        axes[i][0].vlines(start_stim, 0, 60, color='grey')
        axes[i][0].vlines(end_stim, 0, 60, color='grey')
        axes[i][1].plot(t,A_N_A_mean_decrease,'red')
        axes[i][1].plot(t,A_N_A_smooth_decrease,'green')
        axes[i][1].plot(t,A_N_A_mean,'red',alpha=0.6)
        axes[i][1].plot(t,A_N_A_smooth,'green')
        axes[i][1].set_ylim(0,60)
        axes[i][1].vlines(start_stim, 0, 60, color='grey')
        axes[i][1].vlines(end_stim, 0, 60, color='grey')
           

    fig_2c.savefig('figures/'+fig_n+'/'+'Figure2c.png' , bbox_inches='tight')
    plt.close()

