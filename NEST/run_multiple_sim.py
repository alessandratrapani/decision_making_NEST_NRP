import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nest
import nest.raster_plot
import os
from simulate_network import *
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H%M%S")

def run_multiple_sim(results_dir = 'results/standard/', 
                        dt = 0.1,dt_rec = 10.0, 
                        n_trial = 200, mult_coherence = [0.0,0.128,0.512], 
                        order = 400, simtime = 2500.0, start_stim = 200.0,end_stim = 1200.0,
                        stimulus_update_interval = 25,
                        fn_fixed_par = "fixed_parameters.csv", fn_tuned_par = "tuned_parameters.csv", 
                        rec_pop=1.):
   
    winner = np.zeros((len(mult_coherence),2))
    for i,coherence in enumerate(mult_coherence):
        win_A=0
        win_B=0
        delta_s_A_winner = []
        delta_s_B_winner = []
        for j in range(n_trial):    
            nest.ResetKernel()
            nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
            t0 = nest.GetKernelStatus('time')

            results, stimulus_A, stimulus_B, noise_A, noise_B = simulate_network(j,coherence, order , start_stim , end_stim , simtime,stimulus_update_interval, fn_fixed_par, fn_tuned_par, rec_pop)     

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

            
            notes = 'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j)
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
        delta_s_A_winner.to_csv(results_dir+'/coh_0-'+ str(coherence)[2:] +'/delta_s_A_winner.csv')
        delta_s_B_winner.to_csv(results_dir+'/coh_0-'+ str(coherence)[2:] +'/delta_s_B_winner.csv')

    win = {'coherence': mult_coherence, 'pop A win': winner[:,0], 'pop B win': winner[:,1]}
    win = pd.DataFrame(win)
    win.to_csv(results_dir+'winners.csv')
    sim_info = {'n_trial':n_trial, 'sim time':simtime, 'start sim': start_stim, 'end sim': end_stim, 'order':order, 'dt_rec':dt_rec}
    sim_info = pd.DataFrame(sim_info, index = ['value'])
    sim_info.to_csv(results_dir+'sim_info.csv')

    return

def extract_results(path, pop):
       
    events = pd.read_csv(path+'events_pop_'+pop+'.csv')
    frequency = pd.read_csv(path+'frequency.csv')
    stimuli = pd.read_csv(path+'stimuli.csv')

    evs = events['ID neuron pop_'+pop].to_numpy()
    ts = events['event time pop_'+pop].to_numpy()
    t = frequency['time'].to_numpy()
    activity = frequency['activity (Hz) pop_'+pop].to_numpy()
    if pop=='inh':
        stimulus = False
        sum_stimulus = False
    else:
        stimulus = stimuli['stimulus pop '+pop].to_numpy()
        sum_stimulus = stimuli['integral stim pop '+pop].to_numpy()


    return evs, ts, t, activity, stimulus, sum_stimulus
