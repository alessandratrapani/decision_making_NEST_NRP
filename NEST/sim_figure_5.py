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

if not os.path.exists('figures/'+fig_n+'/'):
    os.makedirs('figures/'+fig_n+'/')

if run:
    run_multiple_sim(n_trial = 200, mult_coherence = [0.0,0.512], start_stim = start_stim, end_stim = end_stim,simtime = simtime)


fig_5a, axes = plt.subplots(1, 2,  figsize=(3,3))
fig_5b, axesb = plt.subplots(1, 1,  figsize=(3,3))
n_trial = 200
mult_coherence = [0.0,0.512]
win_pop = 'B_win'
thr_activity = 15

for i,coherence in enumerate(mult_coherence):
    B_N_B_mean = []
    decision_time = []
    dt_string = 'standard/'
    for j in range(n_trial): 
        path = 'results/'+dt_string+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j) +'/'
        if os.path.exists(path):
            evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
            B_N_B_mean.append(B_N_B)  
            axes[i].plot(t[ind_start_stim:ind_end_stim], B_N_B[ind_start_stim:ind_end_stim], color='black', label ='pop B')
            axes[i].hlines(15, start_stim, end_stim, 'grey')
            axes[i].set_ylim(0,30)
            axes[i].set_ylabel("A(t) [Hz]")
            axes[i].set_xlabel("Time [ms]")
            decision_time.append(t[B_N_B >= thr_activity][0])  

    axesb.hist(decision_time, color = [0.5-i*0.3,0.4-i*0.3,0.5-i*0.3], linewidth = 2)

    # axesb[i].set_xlim(0,200)
    # axesb[i].set_ylim(0, n_trial)
    axesb.set_xlabel('Decision time [ms]')
    axesb.set_ylabel('Counts #')

    B_N_B_mean = np.mean(B_N_B_mean,axis=0)
    axes[i].plot(t[ind_start_stim:ind_end_stim], B_N_B_mean[ind_start_stim:ind_end_stim], color='green', label ='pop B')
            
    if save:
        fig_5a.savefig('figures/'+fig_n+'/Figure5A.png' , bbox_inches='tight')
        fig_5b.savefig('figures/'+fig_n+'/Figure5B.png' , bbox_inches='tight')
        plt.close()
    else:
        plt.show()