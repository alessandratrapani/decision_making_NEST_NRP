import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from run_multiple_sim import *
import os
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

figure_4a = True
figure_4b = False
start_stim = 200.0
end_stim = 1200.0
simtime = 2500.0
save=False
run = False
run_1 = False

if not os.path.exists('results/'+fig_n+'/'):
    os.makedirs('results/'+fig_n+'/')

if run:
    run_multiple_sim(n_trial = 1000, mult_coherence = [0.0,0.032,0.064,0.128,0.256,0.512,1.,-0.032,-0.064,-0.128,-0.256,-0.512,-1.], start_stim = start_stim, end_stim = end_stim,simtime = simtime)

if figure_4a:
    dt_string = 'prova'
    results_dir = 'results/'+dt_string+'/'
    winner = pd.read_csv(results_dir+'winners.csv')
    n_trial=1000
    
    coherence_level = winner['coherence'].to_numpy()*100
    pop_A_win = 100*(winner['pop A win'][7:-1].to_numpy())/n_trial
    pop_B_win = 100*(winner['pop B win'][1:7].to_numpy())/n_trial

    fig, ax1 = plt.subplots(1,1,figsize = [5,5])
    ax1.scatter(coherence_level[1:7],pop_A_win,'*', color='red')
    ax1.scatter(coherence_level[1:7],pop_B_win,'*', color='blue')
    ax1.set_xlabel('Coherence level %')
    ax1.set_ylabel('%\ of correct choice')
    ax1.set_xscale("log")
    if save:
        fig.savefig('results/'+fig_n+'/Figure4A.eps' , bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if run_1:
    run_multiple_sim(n_trial = 50, mult_coherence = [0.032,0.064,0.128,0.256,0.032,-0.064,-0.128,-0.256], start_stim = start_stim, end_stim = end_stim,simtime = simtime)

if figure_4b:
    fig,axes = plt.subplots(4,1,figsize = [5,10])
    dt_string = 'stardard/'
    mult_coherence = [0.032,0.064,0.128,0.256]
    for i,coherence in enumerate(mult_coherence):
        mean_activity_wB_sB = np.array([])
        mean_activity_wA_sB = np.array([])
        for j in range(n_trial): 
            #Solid curve black: time course of pop B winning with stimulus on B
            win_pop ='win_B'
            path = 'results/'+dt_string+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j)
            if os.path.exists(path):
                notes, evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
                mean_activity_wB_sB = np.append(mean_activity_wB_sB,B_N_B)

            #Dashed curve orange: time course of pop B losing with stimulus on B
            path = 'results/'+dt_string+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j)
            if os.path.exists(path):
                notes, evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
                mean_activity_wA_sB = np.append(mean_activity_wA_sB,B_N_B)
        
        mean_activity_wB_sB = np.mean(mean_activity_wB_sB)
        mean_activity_wA_sB = np.mean(mean_activity_wA_sB)

        axes[i].plt(t,mean_activity_wB_sB,'black')
        axes[i].plt(t,mean_activity_wA_sB,'orange','-')

    mult_coherence = [-0.032,-0.064,-0.128,-0.256]
    for i,coherence in enumerate(mult_coherence):
        mean_activity_wB_sA = 0
        mean_activity_wA_sA = 0
        for j in range(n_trial): 
            #Dashed curve black: time course of pop B winning with stimulus on A
            win_pop ='win_A'
            path = 'results/'+dt_string+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j)
            if os.path.exists(path):
                notes, evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
                mean_activity_wB_sA = np.append(mean_activity_wB_sA,B_N_B)

            #Solid curve orange: time course of pop B losing with stimulus on A
            win_pop ='win_A'
            path = 'results/'+dt_string+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j)
            if os.path.exists(path):
                notes, evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
                mean_activity_wA_sA = np.append(mean_activity_wA_sA,B_N_B)
        
        mean_activity_wB_sA = np.mean(mean_activity_wB_sA)
        mean_activity_wA_sA = np.mean(mean_activity_wA_sA)
        axes[i].plt(t,mean_activity_wB_sA,'black','-')
        axes[i].plt(t,mean_activity_wA_sA,'orange')
    
    if save:
        fig.savefig('results/'+fig_n+'/Figure4B.eps' , bbox_inches='tight')
        plt.close()
    else:
        plt.show()