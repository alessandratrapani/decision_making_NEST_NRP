import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from simulate_network_revstim import *
from run_multiple_sim import *
import os
# FIGURE7:
# 100 c6.4, t_rev= 100, 200, 300, 500, 600, 700, 800, 900, 1000
# 100 t_rv=1000, c = 3.2, 6.4, 12.8, 25.6, 51.2, 70, 80, 90, 100
# a) plot percentage for different stimulus reversal presentation		b) plot percentage for different strenght stimulus
# c) activity control 							                        d) activity con reverse 6.4 to -80 at t_rev=1000


fig_n = 'Figure7'
run_t_reverse = False
run_stim_reverse = False
save = True
if not os.path.exists('results/'+fig_n+'/'):
    os.makedirs('results/'+fig_n+'/')

dt = 0.1
dt_rec = 10.0
n_trial = 100
start_stim = 200.0
simtime = 2500.0
order = 400
stimulus_update_interval = 25
fn_fixed_par = "fixed_parameters.csv"
fn_tuned_par = "tuned_parameters.csv"
rec_pop=1.

figure7a = True
figure7b = True
figure7c = True

if run_t_reverse:
    coherence = 0.064
    stim_rev = -0.8
    multiple_t_rev = [100+start_stim, 200+start_stim, 300+start_stim, 500+start_stim, 600+start_stim, 700+start_stim, 800+start_stim]
    end_stim = 1200.0
    winner = np.zeros((len(multiple_t_rev),2))
    results_dir = 'results/t_reverse/'
    for i,t_rev in enumerate(multiple_t_rev):
        win_A=0
        win_B=0

        for j in range(n_trial):    
            nest.ResetKernel()
            nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
            t0 = nest.GetKernelStatus('time')

            results, stimulus_A, stimulus_B, noise_A, noise_B = simulate_network(t_rev, stim_rev ,j,coherence, order , start_stim , end_stim , simtime,stimulus_update_interval, fn_fixed_par, fn_tuned_par, rec_pop)     

            smA = nest.GetStatus(results["spike_monitor_A"])[0]
            rmA = nest.GetStatus(results["rate_monitor_A"])[0]	
            smB = nest.GetStatus(results["spike_monitor_B"])[0]
            rmB = nest.GetStatus(results["rate_monitor_B"])[0]          	
            smIn = nest.GetStatus(results["spike_monitor_inh"])[0]
            rmIn = nest.GetStatus(results["rate_monitor_inh"])[0] 

            evsA = smA["events"]["senders"]
            tsA = smA["events"]["times"]
            t = np.arange(0., simtime, dt_rec)
            A_N_A = np.ones((t.size, 1)) * np.nan
            trmA = rmA["events"]["times"]
            trmA = trmA * dt - t0
            bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
            A_N_A = np.histogram(trmA, bins=bins)[0] / order*2 / dt_rec
            A_N_A = A_N_A*1000

            evsB = smB["events"]["senders"]
            tsB = smB["events"]["times"]
            B_N_B = np.ones((t.size, 1)) * np.nan
            trmB = rmB["events"]["times"]
            trmB = trmB * dt - t0
            bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
            B_N_B = np.histogram(trmB, bins=bins)[0] / order*2 / dt_rec
            B_N_B = B_N_B*1000

            evsIn = smIn["events"]["senders"]
            tsIn = smIn["events"]["times"]
            I_N_I = np.ones((t.size, 1)) * np.nan
            trmIn = rmIn["events"]["times"]
            trmIn = trmIn * dt - t0
            bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
            I_N_I = np.histogram(trmIn, bins=bins)[0] / order*1*rec_pop / dt_rec
            I_N_I = I_N_I*1000

            int_stimulus_A = np.zeros((int(simtime)))
            int_stimulus_B = np.zeros((int(simtime)))

            for i in range(1,int(simtime)):
                int_stimulus_A[i] = int_stimulus_A[i-1]+stimulus_A[i]
                int_stimulus_B[i] = int_stimulus_B[i-1]+stimulus_B[i]

            if np.mean(A_N_A[-10:-1])>np.mean(B_N_B[-10:-1]):
                win_A = win_A + 1
                winner[i,0]=win_A
                c = 'red'
                delta_s_A_winner.append(int_stimulus_A[-1] - int_stimulus_B[-1]) 
                win_pop = 'A_win'
            else:
                win_B = win_B + 1
                winner[i,1]=win_B
                c = 'blue'
                delta_s_B_winner.append(int_stimulus_A[-1] - int_stimulus_B[-1])
                win_pop = 'B_win'

            
            notes = 't_rev'+str(t_rev) +'/'+win_pop+ '/trial_'+ str(j)
            saving_dir = results_dir+notes+'/'
            if not os.path.exists(saving_dir):
                os.makedirs(saving_dir)

            raster_A = {'ID neuron pop_A':evsA, 'event time pop_A':tsA}
            raster_B = { 'ID neuron pop_B':evsB, 'event time pop_B':tsB}
            raster_In = { 'ID neuron pop_inh':evsIn, 'event time pop_inh':tsIn}
            activity = {'time':t,'activity (Hz) pop_A': A_N_A, 'activity (Hz) pop_B': B_N_B, 'activity (Hz) pop_inh': I_N_I}
            stimuli = {'stimulus pop A': stimulus_A,'stimulus pop B': stimulus_B, 'integral stim pop A': int_stimulus_A,'integral stim pop B': int_stimulus_B}
            
            events_A = pd.DataFrame(raster_A)
            events_B = pd.DataFrame(raster_B)
            events_inh = pd.DataFrame(raster_In)
            frequency = pd.DataFrame(activity)
            stimuli = pd.DataFrame(stimuli)
            
            events_A.to_csv(saving_dir+'events_pop_A.csv')
            events_B.to_csv(saving_dir+'events_pop_B.csv')
            frequency.to_csv(saving_dir+'frequency.csv')
            events_inh.to_csv(saving_dir+notes+'_events_pop_inh.csv')
            stimuli.to_csv(saving_dir+'stimuli.csv')

        delta_s_A_winner = {'delta_s_A_winner':delta_s_A_winner}
        delta_s_B_winner = {'delta_s_B_winner':delta_s_B_winner}
        delta_s_A_winner = pd.DataFrame(delta_s_A_winner)
        delta_s_B_winner = pd.DataFrame(delta_s_B_winner)
        delta_s_A_winner.to_csv(results_dir+'/t_rev'+str(t_rev) +'/delta_s_A_winner.csv')
        delta_s_B_winner.to_csv(results_dir+'/t_rev'+str(t_rev) +'/delta_s_B_winner.csv')

    win = {'coherence': multiple_t_rev, 'pop A win': winner[:,0], 'pop B win': winner[:,1]}
    win = pd.DataFrame(win)
    win.to_csv(results_dir+'winners.csv')
    sim_info = {'n_trial':n_trial, 'sim time':simtime, 'start sim': start_stim, 'end sim': end_stim, 'order':order, 'dt_rec':dt_rec}
    sim_info = pd.DataFrame(sim_info, index = ['value'])
    sim_info.to_csv(results_dir+'sim_info.csv')

if run_stim_reverse:
    coherence = 0.064
    t_rev = 1200.0
    multiple_stim_rev = [-0.032,-0.064,-0.128,-0.256,-0.512,-0.7,-0.8,-1.0]
    end_stim = 2200.0
    winner = np.zeros((len(multiple_stim_rev),2))
    results_dir = 'results/stim_reverse/'
    for i,stim_rev in enumerate(multiple_stim_rev):
        win_A=0
        win_B=0

        for j in range(n_trial):    
            nest.ResetKernel()
            nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
            t0 = nest.GetKernelStatus('time')

            results, stimulus_A, stimulus_B, noise_A, noise_B = simulate_network(t_rev, stim_rev ,j,coherence, order , start_stim , end_stim , simtime,stimulus_update_interval, fn_fixed_par, fn_tuned_par, rec_pop)     

            smA = nest.GetStatus(results["spike_monitor_A"])[0]
            rmA = nest.GetStatus(results["rate_monitor_A"])[0]	
            smB = nest.GetStatus(results["spike_monitor_B"])[0]
            rmB = nest.GetStatus(results["rate_monitor_B"])[0]          	
            smIn = nest.GetStatus(results["spike_monitor_inh"])[0]
            rmIn = nest.GetStatus(results["rate_monitor_inh"])[0] 

            evsA = smA["events"]["senders"]
            tsA = smA["events"]["times"]
            t = np.arange(0., simtime, dt_rec)
            A_N_A = np.ones((t.size, 1)) * np.nan
            trmA = rmA["events"]["times"]
            trmA = trmA * dt - t0
            bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
            A_N_A = np.histogram(trmA, bins=bins)[0] / order*2 / dt_rec
            A_N_A = A_N_A*1000

            evsB = smB["events"]["senders"]
            tsB = smB["events"]["times"]
            B_N_B = np.ones((t.size, 1)) * np.nan
            trmB = rmB["events"]["times"]
            trmB = trmB * dt - t0
            bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
            B_N_B = np.histogram(trmB, bins=bins)[0] / order*2 / dt_rec
            B_N_B = B_N_B*1000

            evsIn = smIn["events"]["senders"]
            tsIn = smIn["events"]["times"]
            I_N_I = np.ones((t.size, 1)) * np.nan
            trmIn = rmIn["events"]["times"]
            trmIn = trmIn * dt - t0
            bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
            I_N_I = np.histogram(trmIn, bins=bins)[0] / order*1*rec_pop / dt_rec
            I_N_I = I_N_I*1000

            int_stimulus_A = np.zeros((int(simtime)))
            int_stimulus_B = np.zeros((int(simtime)))

            for i in range(1,int(simtime)):
                int_stimulus_A[i] = int_stimulus_A[i-1]+stimulus_A[i]
                int_stimulus_B[i] = int_stimulus_B[i-1]+stimulus_B[i]

            if np.mean(A_N_A[-10:-1])>np.mean(B_N_B[-10:-1]):
                win_A = win_A + 1
                winner[i,0]=win_A
                c = 'red'
                delta_s_A_winner.append(int_stimulus_A[-1] - int_stimulus_B[-1]) 
                win_pop = 'A_win'
            else:
                win_B = win_B + 1
                winner[i,1]=win_B
                c = 'blue'
                delta_s_B_winner.append(int_stimulus_A[-1] - int_stimulus_B[-1])
                win_pop = 'B_win'

            
            notes = 'stim_rev'+ str(stim_rev)+'/'+win_pop+ '/trial_'+ str(j)
            saving_dir = results_dir+notes+'/'
            if not os.path.exists(saving_dir):
                os.makedirs(saving_dir)

            raster_A = {'ID neuron pop_A':evsA, 'event time pop_A':tsA}
            raster_B = { 'ID neuron pop_B':evsB, 'event time pop_B':tsB}
            raster_In = { 'ID neuron pop_inh':evsIn, 'event time pop_inh':tsIn}
            activity = {'time':t,'activity (Hz) pop_A': A_N_A, 'activity (Hz) pop_B': B_N_B, 'activity (Hz) pop_inh': I_N_I}
            stimuli = {'stimulus pop A': stimulus_A,'stimulus pop B': stimulus_B, 'integral stim pop A': int_stimulus_A,'integral stim pop B': int_stimulus_B}
            
            events_A = pd.DataFrame(raster_A)
            events_B = pd.DataFrame(raster_B)
            events_inh = pd.DataFrame(raster_In)
            frequency = pd.DataFrame(activity)
            stimuli = pd.DataFrame(stimuli)
            
            events_A.to_csv(saving_dir+'events_pop_A.csv')
            events_B.to_csv(saving_dir+'events_pop_B.csv')
            frequency.to_csv(saving_dir+'frequency.csv')
            events_inh.to_csv(saving_dir+notes+'_events_pop_inh.csv')
            stimuli.to_csv(saving_dir+'stimuli.csv')

        delta_s_A_winner = {'delta_s_A_winner':delta_s_A_winner}
        delta_s_B_winner = {'delta_s_B_winner':delta_s_B_winner}
        delta_s_A_winner = pd.DataFrame(delta_s_A_winner)
        delta_s_B_winner = pd.DataFrame(delta_s_B_winner)
        delta_s_A_winner.to_csv(results_dir+'/stim_rev'+ str(stim_rev)+'/delta_s_A_winner.csv')
        delta_s_B_winner.to_csv(results_dir+'/stim_rev'+ str(stim_rev)+'/delta_s_B_winner.csv')

    win = {'coherence': multiple_stim_rev, 'pop A win': winner[:,0], 'pop B win': winner[:,1]}
    win = pd.DataFrame(win)
    win.to_csv(results_dir+'winners.csv')
    sim_info = {'n_trial':n_trial, 'sim time':simtime, 'start sim': start_stim, 'end sim': end_stim, 'order':order, 'dt_rec':dt_rec}
    sim_info = pd.DataFrame(sim_info, index = ['value'])
    sim_info.to_csv(results_dir+'sim_info.csv')

if figure7a:
    save=False
    dt_string = 'prova'
    results_dir = 'results/t_reverse/'
    winner = pd.read_csv(results_dir+'winners.csv')
    n_trial=1000
    
    coherence_level = winner['coherence'].to_numpy()*100
    pop_A_win = 100*(winner['pop A win'].to_numpy())/n_trial
    pop_B_win = 100*(winner['pop B win'].to_numpy())/n_trial

    fig, ax1 = plt.subplots(1,1,figsize = [5,5])
    ax1.scatter(coherence_level,pop_A_win,'*', color='red')
    ax1.scatter(coherence_level,pop_B_win,'*', color='blue')
    ax1.set_xlabel('Coherence level %')
    ax1.set_ylabel('%\ of correct choice')
    ax1.set_xscale("log")
    if save:
        fig.savefig('results/'+fig_n+'/Figure7A.eps' , bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if figure7b:
    save=False
    dt_string = 'prova'
    results_dir = 'results/stim_reverse/'
    winner = pd.read_csv(results_dir+'winners.csv')
    n_trial=1000
    
    coherence_level = winner['coherence'].to_numpy()*100
    pop_A_win = 100*(winner['pop A win'].to_numpy())/n_trial
    pop_B_win = 100*(winner['pop B win'].to_numpy())/n_trial

    fig, ax1 = plt.subplots(1,1,figsize = [5,5])
    ax1.scatter(coherence_level,pop_A_win,'*', color='red')
    ax1.scatter(coherence_level,pop_B_win,'*', color='blue')
    ax1.set_xlabel('Coherence level %')
    ax1.set_ylabel('%\ of correct choice')
    ax1.set_xscale("log")
    if save:
        fig.savefig('results/'+fig_n+'/Figure7B.eps' , bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if figure7c:
    fig,axes = plt.subplots(1,2,figsize = [5,10])

    dt_string = 'prova'
    coherence = 0.064
    mean_activity_wB_sB = np.array([])
    mean_activity_wA_sB = np.array([])
    for j in range(n_trial): 
        #Solid curve black: time course of pop B winning with stimulus on B
        win_pop ='win_B'
        path = 'results/'+dt_string+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j)
        if os.path.exists(path):
            notes, evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
            axes[0].plt(t,B_N_B,'black')
            mean_activity_wB_sB = np.append(mean_activity_wB_sB,B_N_B)    
    mean_activity_wB_sB = np.mean(mean_activity_wB_sB)
    axes[0].plt(t,mean_activity_wB_sB,'green')

    
    coherence = 0.064
    mean_activity_wB_sB = np.array([])
    mean_activity_wA_sB = np.array([])
    stim_rev=-0.800
    for j in range(n_trial): 
        #Solid curve black: time course of pop B winning with stimulus on B
        win_pop ='win_B'
        path = 'results/t_reverse/stim_rev'+ str(stim_rev)+'/'+win_pop+ '/trial_'+ str(j)
        if os.path.exists(path):
            notes, evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
            axes[1].plt(t,B_N_B,'black')
            mean_activity_wB_sB = np.append(mean_activity_wB_sB,B_N_B)    
    mean_activity_wB_sB = np.mean(mean_activity_wB_sB)
    axes[1].plt(t,mean_activity_wB_sB,'green')
    if save:
        fig.savefig('results/'+fig_n+'/Figure7C.eps' , bbox_inches='tight')
        plt.close()
    else:
        plt.show()