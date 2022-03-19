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
n_trial = 100

if not os.path.exists('figures/'+fig_n+'/'):
    os.makedirs('figures/'+fig_n+'/')

if run:
    for d in end_stim:
        results_dir = 'results/stim_end_'+str(d)+'/'
        run_multiple_sim(results_dir =results_dir ,n_trial = n_trial, mult_coherence = [0.032, 0.064 , 0.128, 0.256, 0.512], start_stim = start_stim, end_stim = d, simtime = simtime)

fig, ax1 = plt.subplots(1,1,figsize = [5,5])
marker=['o','D','s','X','^']
for n,d in enumerate(end_stim):
    results_dir = 'results/stim_end_'+str(d)+'/'
    winner = pd.read_csv(results_dir+'winners.csv')
    coherence_level = winner['coherence'].to_numpy()*100
    pop_B_win = 100*(winner['pop B win'].to_numpy())/n_trial 
    ax1.semilogx(coherence_level,pop_B_win,marker=marker[n], label = 'T='+str(d)+' ms')
    ax1.set_xlabel('Coherence level %')
    ax1.set_ylabel('%\ of correct choice')


dt_string = '2022-03-15_233244'
results_dir = 'results/'+dt_string+'/'
winner = pd.read_csv(results_dir+'2022-03-15_233244_winners.csv')
n_trial=1000

#coherence_level = winner['coherence'].to_numpy()*100
pop_B_win = 100*(winner['pop B win'].to_numpy())/n_trial
d=1000
ax1.semilogx(coherence_level,pop_B_win[1:6],'black',marker='*',label = 'T='+str(d)+' ms')
ax1.legend()
ax1.set_xlim(2*1e0, 2*1e2)

    

if save:
    fig.savefig('figures/'+fig_n+'/Figure6.png' , bbox_inches='tight')
    plt.close()
else:
    plt.show()