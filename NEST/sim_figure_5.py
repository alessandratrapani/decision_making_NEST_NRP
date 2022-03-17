import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from run_multiple_sim import *
import os
# FIGURE5: comparison reaction time
# sim 200 0.0, 200 51.2
# a) activity 0.0 (intorno stimulus)	activity 51.2 (intorno stimulus)
# b) hist reaction time 0.0		hist reaction time 51.2

fig_n = 'Figure5'
run = False
save = True
start_stim = 200.0
end_stim = 1200.0
simtime = 2500.0
dt_rec = 10.0
ind_start_stim = int(start_stim/dt_rec)
ind_end_stim = int(end_stim/dt_rec)

if not os.path.exists('results/'+fig_n+'/'):
    os.makedirs('results/'+fig_n+'/')

if run:
    run_multiple_sim(n_trial = 200, mult_coherence = [0.0,0.512], start_stim = start_stim, end_stim = end_stim,simtime = simtime)


fig_5a, axes = plt.subplots(1, 2,  figsize=(3,3))
fig_5b, axesb = plt.subplots(1, 2,  figsize=(3,3))
n_trial = 200
mult_coherence = [0.0,0.512]
win_pop = 'win_B'
thr_activity = 15

for i,coherence in enumerate(mult_coherence):
    B_N_B_mean = np.array([])
    decision_time = np.array([])
    dt_string = 'standard/'
    for j in range(n_trial): 
        path = 'results/'+dt_string+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j)
        if os.path.exists(path):
            notes, evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
            B_N_B_mean = np.append(B_N_B_mean,B_N_B,axis=0)  
            axes[i].plot(t[ind_start_stim:ind_end_stim], B_N_B[ind_start_stim:ind_end_stim], color='black', label ='pop B')
            axes[i].hlines(15, start_stim, end_stim, 'grey')
            axes[i].set_ylim(0,30)
            axes[i].set_ylabel("A(t) [Hz]")
            axes[i].set_xlabel("Time [ms]")
            decision_time=np.append(decision_time,t[B_N_B >= thr_activity][0], axis=0)  

    axesb[i].hist(decision_time, histtype = 'step', color = 'black', linewidth = 2)
    axesb[i].set_xlim(0,200)
    axesb[i].set_ylim(0, n_trial)
    axesb[i].set_xlabel('Decision time [ms]')
    axesb[i].set_ylabel('Counts #')

    B_N_B_mean = np.mean(B_N_B_mean)
    axes[i].plot(t[ind_start_stim:ind_end_stim], B_N_B_mean[ind_start_stim:ind_end_stim], color='green', label ='pop B')
            
    if save:
        fig_5a.savefig('results/'+fig_n+'/Figure5A.eps' , bbox_inches='tight')
        fig_5b.savefig('results/'+fig_n+'/Figure5B.eps' , bbox_inches='tight')
        plt.close()
    else:
        plt.show()