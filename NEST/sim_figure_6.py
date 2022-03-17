import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from run_multiple_sim import *
import os
# FIGURE6: stimulus presentation
# sim 100 c3.2, c6.4 , c12.8, c25.6, c51.2 x T300 ms, T500, T700, T1000
# plot percentage + holdon for each time stimulus duration
fig_n = 'Figure6'
run = False
save = True
start_stim = 200.0
end_stim = [300.0+start_stim,500.0+start_stim,700.0+start_stim]
simtime = 2500.0

if not os.path.exists('results/'+fig_n+'/'):
    os.makedirs('results/'+fig_n+'/')

if run:
    for d in end_stim:
        results_dir = 'results/stim_end_'+str(d)+'/'
        run_multiple_sim(results_dir =results_dir ,n_trial = 50, mult_coherence = [0.032, 0.064 , 0.128, 0.256, 0.512], start_stim = start_stim, end_stim = d, simtime = simtime)


save=False
fig, ax1 = plt.subplots(1,1,figsize = [5,5])
marker=['o','D','s','X','^']
for n,d in enumerate(end_stim):
    results_dir = 'results/stim_end_'+str(d)+'/'
    winner = pd.read_csv(results_dir+'winners.csv')
    n_trial=50
    coherence_level = winner['coherence'].to_numpy()*100
    pop_B_win = 100*(winner['pop B win'].to_numpy())/n_trial   
    ax1.scatter(coherence_level[1:7],pop_B_win,marker=marker[n], label = 'T='+str(d)+' ms')
    ax1.set_xlabel('Coherence level %')
    ax1.set_ylabel('%\ of correct choice')
    ax1.set_xscale("log")

dt_string = 'prova'
results_dir = 'results/'+dt_string+'/'
winner = pd.read_csv(results_dir+'winners.csv')
n_trial=1000

coherence_level = winner['coherence'].to_numpy()*100
pop_B_win = 100*(winner['pop B win'][1:7].to_numpy())/n_trial
fig, ax1 = plt.subplots(1,1,figsize = [5,5])
ax1.scatter(coherence_level[1:7],pop_B_win,'*', 'T=1000 ms')
ax1.set_xlabel('Coherence level %')
ax1.set_ylabel('%\ of correct choice')
ax1.set_xscale("log")
    

if save:
    fig.savefig('results/'+fig_n+'/Figure6.eps' , bbox_inches='tight')
    plt.close()
else:
    plt.show()