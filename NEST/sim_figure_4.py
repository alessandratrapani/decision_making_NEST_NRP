import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from run_multiple_sim import *
import os
import scipy.signal as signal
# FIGURE4: percentage of pop choosing correctly
# a) Comparison with Wang2002 directly
# plot square Wang percentage / dot my percentage + holdon Weibull function  
# %fit with parameters, are the parameters in accordance with Wang 2002 and previous studies? The percentage response for A can be fitted by \% correct 1-epsilon*exp(-(c/alpha)**beta) and is 1-epsilon at zero coherence, so epsilon=0.5. 
# b) Time evolution of the ramping activity for different coherence level: 
# solid curve black: time course of pop B winning with stimulus on B
# dashed curve black: time course of pop B winning when stimulus on A
# solid curve orange: time course of pop B losing with stimulus on A
# dashed curve orange: time course of pop B losing with stimulus on B
# \\c 3.2
# \\c 6.4, 
# \\c 12.8
# \\c 25.6 

fig_n = 'Figure4'

figure_4a = False
figure_4b = True
start_stim = 200.0
end_stim = 1200.0
simtime = 2500.0
save=True
run = False
run_1 = False

if not os.path.exists('figures/'+fig_n+'/'):
    os.makedirs('figures/'+fig_n+'/')

if run:
    run_multiple_sim(n_trial = 1000, mult_coherence = [0.0,0.032,0.064,0.128,0.256,0.512,1.,-0.032,-0.064,-0.128,-0.256,-0.512,-1.], start_stim = start_stim, end_stim = end_stim,simtime = simtime)

if figure_4a:
    dt_string = '2022-03-15_233244'
    results_dir = 'results/'+dt_string+'/'
    winner = pd.read_csv(results_dir+dt_string+'_winners.csv')
    n_trial=1000
    
    coherence_level = [3.2,6.4,12.8,25.6,51.2,100.]
    pop_A_win = 100*(winner['pop A win'].to_numpy())/n_trial
    pop_B_win = 100*(winner['pop B win'].to_numpy())/n_trial

    fig, ax1 = plt.subplots(1,1,figsize = [5,5])
    ax1.semilogx(coherence_level,pop_A_win[8:],'s-', color='red')
    ax1.semilogx(coherence_level,pop_B_win[:6],'o-', color='blue')
    ax1.set_xlim(2*1e0, 2*1e2)
    ax1.set_xlabel('Coherence level %')
    ax1.set_ylabel('%\ of correct choice')
    if save:
        fig.savefig('figures/'+fig_n+'/Figure4A.png' , bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if run_1:
    run_multiple_sim(n_trial = 50, mult_coherence = [0.032,0.064,0.128,0.256,0.032,-0.064,-0.128,-0.256], start_stim = start_stim, end_stim = end_stim,simtime = simtime)

if figure_4b:
    fig,ax4b = plt.subplots(4,1,figsize = [5,10])
    dt_string = 'standard/'
    #DA SETTARE
    mult_coherence = [0.0,0.032, 0.064, 0.128]
    n_trial = 200
    for i,coherence in enumerate(mult_coherence):
        mean_activity_wB_sB = []
        mean_activity_wA_sB = []
        for j in range(n_trial): 
            #Solid curve black: time course of pop B winning with stimulus on B
            win_pop ='B_win'
            path = 'results/'+dt_string+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j) + '/'

            if os.path.exists(path):
                evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
                mean_activity_wB_sB.append(B_N_B)

            #Dashed curve orange: time course of pop B losing with stimulus on B
            win_pop ='A_win'
            path = 'results/'+dt_string+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j) +'/'

            if os.path.exists(path):
                evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
                mean_activity_wA_sB.append(B_N_B)
            
        mean_activity_wB_sB = np.mean(mean_activity_wB_sB,axis=0)
        mean_activity_wA_sB = np.mean(mean_activity_wA_sB,axis=0)
        mean_activity_wB_sB = signal.medfilt(mean_activity_wB_sB,35)
        mean_activity_wA_sB = signal.medfilt(mean_activity_wA_sB,35)
        
        ax4b[i].plot(t[0:120],mean_activity_wB_sB[0:120],'black')
        ax4b[i].plot(t[0:120],mean_activity_wA_sB[0:120],'orange','--',alpha=0.6)

    #DA SETTARE
    mult_coherence = [0.0,0-.032, -0.064, -0.128]
    for i,coherence in enumerate(mult_coherence):
        mean_activity_wB_sA = []
        mean_activity_wA_sA = []
        for j in range(n_trial): 
            #Dashed curve black: time course of pop B winning with stimulus on A
            win_pop ='B_win'
            path = 'results/'+dt_string+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j)+ '/'
   
            if os.path.exists(path):
                evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
                mean_activity_wB_sA.append(B_N_B)
                

            #Solid curve orange: time course of pop B losing with stimulus on A
            win_pop ='A_win'
            path = 'results/'+dt_string+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j)+ '/'
       
            if os.path.exists(path):
                evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
                mean_activity_wA_sA.append(B_N_B)
        
        mean_activity_wB_sA = np.mean(mean_activity_wB_sA,axis=0)
        mean_activity_wA_sA = np.mean(mean_activity_wA_sA,axis=0)
        mean_activity_wB_sA = signal.medfilt(mean_activity_wB_sA,35)
        mean_activity_wA_sA = signal.medfilt(mean_activity_wA_sA,35)
        
        ax4b[i].plot(t[0:120],mean_activity_wB_sA[0:120],'black', '--',alpha=0.6)
        ax4b[i].plot(t[0:120],mean_activity_wA_sA[0:120],'orange')
    
        ax4b[i].set_ylim(0,40)
    if save:
        fig.savefig('figures/'+fig_n+'/Figure4B.png' , bbox_inches='tight')
        plt.close()
    else:
        plt.show()