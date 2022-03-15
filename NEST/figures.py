import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
save = True
dt_string = '2022-03-15_150803'
coherence = 0.512
trial = 5

results_directory = os.getcwd()+'/results/'+dt_string+'/'

sim_info = pd.read_csv(results_directory+dt_string+'_sim_info.csv')
simtime = sim_info['sim time'].to_numpy()
start_stim = sim_info['start sim'].to_numpy()
end_stim = sim_info['end sim'].to_numpy()
order = sim_info['order'].to_numpy()
n_trial = sim_info['n_trial'].to_numpy()
#dt_rec = sim_info['dt_rec'].to_numpy()
dt_rec = 10

def extract_results(dt_string, coherence, trial, pop):
    results_directory = os.getcwd()+'/results/'+dt_string+'/'
    notes = 'coh_' + '0-'+ str(coherence)[2:] + '_trial_'+ str(trial)
    trial_directory = results_directory + notes +'/'
    
    events = pd.read_csv(trial_directory+notes+'_events_pop_'+pop+'.csv')
    evs = events['ID neuron pop_'+pop].to_numpy()
    ts = events['event time pop_'+pop].to_numpy()
    frequency = pd.read_csv(trial_directory+notes+'_frequency.csv')
    t = frequency['time'].to_numpy()
    activity = frequency['activity (Hz) pop_'+pop].to_numpy()
    stimuli = pd.read_csv(trial_directory+notes+'_stimuli.csv')
    stimulus = stimuli['stimulus pop '+pop].to_numpy()
    sum_stimulus = stimuli['integral stim pop '+pop].to_numpy()

    return notes, evs, ts, t, activity, stimulus, sum_stimulus

figure_1 = True
figure_2a = True
figure_2b = True
figure_2c = True
figure_5 = True
# %% FIGURE_1 
#0.0 t4-8
#0.128 t3-4-6-7-9(best)
#0.512 t1-3-5(best)
#-0.512 t3-5
#-0.128 t0-1
if figure_1:
    fig= None
    ax_raster = None
    ax_rate = None
    fig, ((ax_raster_A,ax_raster_B), (ax_rate_A,ax_rate_B)) = plt.subplots(2, 2, sharex=True,  figsize=(5,3))
    plt.suptitle('Coherence ' + str(coherence*100) + '%')

    notes, evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(dt_string, coherence, trial, 'A')
    ax_raster_A.scatter(tsA, evsA,marker = '|', linewidths = 0.8, color='red', label ='pop A')
    ax_raster_A.set_ylabel("neuron #")
    ax_raster_A.vlines(start_stim, 0, 800, color='grey')
    ax_raster_A.vlines(end_stim, 0, 800, color='grey')
    ax_raster_A.set_title("Raster Plot pop A ", fontsize=10)
    ax_rate_A.plot(t, A_N_A, color='red', label ='pop A')
    ax_rate_A.fill_between(t, A_N_A, color='red')
    ax_rate_A.vlines(start_stim, 0, 40, color='grey')
    ax_rate_A.vlines(end_stim, 0, 40, color='grey')
    ax_rate_A.set_ylabel("A(t) [Hz]")
    ax_rate_A.set_title("Activity Pop A", fontsize=10)

    notes, evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(dt_string, coherence, trial, 'B')
    ax_raster_B.scatter(tsB, evsB,marker= '|',  linewidths = 0.8,color='blue', label ='pop B')
    ax_raster_B.set_ylabel("neuron #")
    ax_raster_B.set_title("Raster Plot pop B ", fontsize=10)
    ax_raster_B.vlines(start_stim, 800, 1600, color='grey')
    ax_raster_B.vlines(end_stim, 800, 1600, color='grey')
    ax_rate_B.plot(t, B_N_B, color='blue', label ='pop B')
    ax_rate_B.fill_between(t, B_N_B, color='blue')
    ax_rate_B.vlines(start_stim, 0, 40, color='grey')
    ax_rate_B.vlines(end_stim, 0, 40, color='grey')
    ax_rate_B.set_ylabel("A(t) [Hz]")
    ax_rate_B.set_title("Activity Pop B", fontsize=10)
    plt.xlabel("t [ms]")

    if save:
        if not os.path.exists(os.getcwd()+'/figures/'):
            os.makedirs(os.getcwd()+'/figures/')
        fig.savefig(os.getcwd()+'/figures/Figure1.eps' , bbox_inches='tight')
        plt.close()
    else:
        plt.show() 

# %%FIGURE_2
#FIGURE2A
if figure_2a:
    fig = None
    ax_raster = None
    ax_rate = None
    fig, ((ax_raster_A,ax_raster_B), (ax_rate_A,ax_rate_B), (ax_stimuli_A,ax_stimuli_B), (ax_sum_stimuli_A,ax_sum_stimuli_B)) = plt.subplots(4, 2, sharex=True, figsize=(16,9))
    
    coherence = 0.0
    plt.suptitle('Coherence ' + str(coherence*100) + '%')

    trial = 8
    notes, evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(dt_string, coherence, trial, 'A')
    notes, evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(dt_string, coherence, trial, 'B')
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

    trial = 3
    notes, evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(dt_string, coherence, trial, 'A')
    notes, evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(dt_string, coherence, trial, 'B')
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
        if not os.path.exists(os.getcwd()+'/figures/' ):
            os.makedirs(os.getcwd()+'/figures/' )
        fig.savefig(os.getcwd()+'/figures/Figure2A.eps' , bbox_inches='tight')
        plt.close()
    else:
        plt.show() 
    
# FIGURE2B
if figure_2b:
    fig, decisional_space = plt.subplots(1,1,figsize = [5,5])

    trial = 8
    notes, evsA, tsA, t, A_N_A_1, stimulus_A, sum_stimulus_A = extract_results(dt_string, coherence, trial, 'A')
    notes, evsB, tsB, t, B_N_B_1, stimulus_B, sum_stimulus_B = extract_results(dt_string, coherence, trial, 'B')

    if np.mean(A_N_A_1[-10:-1])>np.mean(B_N_B_1[-10:-1]):
        c='red'
        winner = 'pop_A'
    else:
        winner = 'pop_B'
        c='blue'
    decisional_space.plot(A_N_A_1,B_N_B_1, color=c)
    decisional_space.plot([0,40],[0,40], color='grey')
    
    trial = 3
    notes, evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(dt_string, coherence, trial, 'A')
    notes, evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(dt_string, coherence, trial, 'B')
    
    if np.mean(A_N_A[-10:-1])>np.mean(B_N_B[-10:-1]):
        c='red'
        winner = 'pop_A'
    else:
        winner = 'pop_B'
        c='blue'
    decisional_space.plot(A_N_A,B_N_B, color=c)

    decisional_space.set_xlim(-0.1,40)
    decisional_space.set_ylim(-0.1,40)
    decisional_space.set_xlabel("Firing rate pop A (Hz)")
    decisional_space.set_ylabel("Firing rate pop B (Hz)")
    decisional_space.set_title("Decision Space")
    if save:
        if not os.path.exists(os.getcwd()+'/figures/' ):
            os.makedirs(os.getcwd()+'/figures/' )
        fig.savefig(os.getcwd()+'/figures/Figure2B.eps' , bbox_inches='tight')
        plt.close()
    else:
        plt.show()

#FIGURE2C
if figure_2c:
    
    delta_s_A_winner = np.array([])
    delta_s_B_winner = np.array([])
    coherence = 0.0
    fig, ax1 = plt.subplots(1,1,figsize = [5,5])
    for j in range(int(n_trial)): 
        notes, evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(dt_string, coherence, j, 'A')
        notes, evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(dt_string, coherence, j, 'B')
        
        if np.mean(A_N_A[-10:-1])>np.mean(B_N_B[-10:-1]):
            c='red'
            delta_s_A_winner = np.append(delta_s_A_winner, [sum_stimulus_A[-1] - sum_stimulus_B[-1]])
            winner = 'pop_A'
        else:
            c='blue'
            delta_s_B_winner = np.append(delta_s_B_winner, [sum_stimulus_A[-1] - sum_stimulus_B[-1]])
            winner = 'pop_B'
            
    ax1.hist(delta_s_A_winner, histtype = 'step', color = 'red', linewidth = 2)
    ax1.hist(delta_s_B_winner, histtype = 'step', color = 'blue', linewidth = 2)
    ax1.set_xlabel('Time integral of $s_1$(t) - $s_2$(t)')
    ax1.set_ylabel('Count #')
    if save:
        if not os.path.exists(os.getcwd()+'/figures/' ):
            os.makedirs(os.getcwd()+'/figures/' )
        fig.savefig(os.getcwd()+'/figures/Figure2C.eps' , bbox_inches='tight')
        plt.close()
    else:
        plt.show()
#TODO 5 coin toss with 0 variability in the inputs

#TODO 6 network performance and error trials Fig4a
#TODO 7 evaluate the network time courses at: 3.2%, 6.4%,12.8%, 25.6% Fig4b (n trials=1000 and take the mean)

#%% FIGURE_5
# compare 0.0% and 51.2% --> time that it take to cross the 15Hz threshold (figure 5a-b) --> need to find a linear relationship between mean reaction time and log coherence level
if figure_5:
    
    fig, ((rate_51,rate_0),(counts_51,counts_0)) = plt.subplots(2,2,figsize = [5,5])
    decision_time_51=np.array([])
    decision_time_0=np.array([])
    for j in range(int(n_trial)):        
        coherence = -0.512        
        notes, evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(dt_string, coherence, j, 'A')
        notes, evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(dt_string, coherence, j, 'B')
        ind_start_stim = int(start_stim/dt_rec)
        ind_end_stim = int(end_stim/dt_rec)
        if np.mean(A_N_A[-10:-1])>np.mean(B_N_B[-10:-1]):
            rate_51.plot(t[ind_start_stim:ind_end_stim], A_N_A[ind_start_stim:ind_end_stim], color='black', label ='pop A')
            rate_51.hlines(15, start_stim, end_stim, 'grey')
            rate_51.set_ylim(0,30)
            rate_51.set_ylabel("A(t) [Hz]")
            rate_51.set_xlabel("Time [ms]")
            decision_time_51=np.append(decision_time_51,t[A_N_A >= 15][0])

        coherence = 0.0
        notes, evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(dt_string, coherence, j, 'A')
        notes, evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(dt_string, coherence, j, 'B')
        if np.mean(A_N_A[-10:-1])>np.mean(B_N_B[-10:-1]):
            rate_0.plot(t[ind_start_stim:ind_end_stim], A_N_A[ind_start_stim:ind_end_stim], color='black', label ='pop A')
            rate_0.hlines(15, start_stim, end_stim, 'grey')
            rate_0.set_ylim(0,30)
            rate_0.set_ylabel("A(t) [Hz]")
            rate_0.set_xlabel("Time [ms]")
            decision_time_0=np.append(decision_time_0,t[A_N_A >= 15][0])
    counts_51.hist(decision_time_51, histtype = 'step', color = 'black', linewidth = 2)
    counts_51.set_xlim(0,1000)
    counts_51.set_ylim(0, n_trial)
    counts_51.set_xlabel('Decision time [ms]')
    counts_51.set_ylabel('Counts #')
    counts_0.hist(decision_time_0, histtype = 'step', color = 'black', linewidth = 2)
    counts_0.set_xlim(0,1000)
    counts_0.set_ylim(0, n_trial)
    counts_0.set_xlabel('Decision time [ms]')
    counts_0.set_ylabel('Counts #')

    if save:
        if not os.path.exists(os.getcwd()+'/figures/' ):
            os.makedirs(os.getcwd()+'/figures/' )
        fig.savefig(os.getcwd()+'/figures/Figure5.eps' , bbox_inches='tight')
        plt.close()
    else:
        plt.show()


#TODO 9 test stimulus duration fig6A

#TODO 10 test persistent activity --> decrease recurrent exc weights -> again 0 12.8 51.2
# da notare: no ramping, no winner takes all a 0.0, no persistent activity.

#TODO 11 testare NMDA slow reverberation --> switch off?

#TODO 12 reverse decision --> possibilità di cambiare quando avviene lo stimolo reverse (Percentage choices for A and B as function of the onset time of reversal.Fig8A Even when the signal is reversed 1 s after the stimulus onset, the decision is still re- versable by a more powerful input. Percent- age choices for A and B as function of the coherence level of the reversed signalFig8B
#coherence above 70%–80%
