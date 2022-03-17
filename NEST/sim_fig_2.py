import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nest
import nest.raster_plot
import os
from simulate_network import *
# FIGURE2: mostrare effetto dei pesi excitatory recurrent
# sim 1 c51.2 decrease, 1 c0.0 decrease, 100 6.4 increase, 1 c51.2 noNMDA
fig_n = 'Figure2'

figure_2a = True
figure_2b = False
figure_2c = True

order = 400
simtime = 3000.0
start_stim = 200.0
end_stim = 1200.0
dt = 0.1
dt_rec = 10.0
rec_pop = 0.5


if figure_2a:
    fig_2a, ((ax_rate_A_0,ax_rate_B_0),(ax_rate_A_51,ax_rate_B_51)) = plt.subplots(2, 2, sharex=True,  figsize=(5,3))
    notes = 'decrease_weight'
    saving_dir = 'results/'+fig_n+'/'+notes+'/'
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    coherence = 0.0

    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
    t0 = nest.GetKernelStatus('time')
    results, stimulus_A, stimulus_B, noise_A, noise_B = simulate_network(1,coherence, order , start_stim , end_stim , simtime, fn_tuned_par = "w_decrease.csv")     
    
    smA = nest.GetStatus(results["spike_monitor_A"])[0]
    rmA = nest.GetStatus(results["rate_monitor_A"])[0]	
    smB = nest.GetStatus(results["spike_monitor_B"])[0]
    rmB = nest.GetStatus(results["rate_monitor_B"])[0] 

    evsA = smA["events"]["senders"]
    tsA = smA["events"]["times"]
    t = np.arange(0., simtime, dt_rec)
    A_N_A = np.ones((t.size, 1)) * np.nan
    trmA = rmA["events"]["times"]
    trmA = trmA * dt - t0
    bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
    A_N_A = np.histogram(trmA, bins=bins)[0] / order*2 / dt_rec
    A_N_A = A_N_A*1000
    ax_rate_A_0.plot(t, A_N_A, color='red', label ='pop A')
    ax_rate_A_0.vlines(start_stim, 0, 40, color='grey')
    ax_rate_A_0.vlines(end_stim, 0, 40, color='grey')
    ax_rate_A_0.set_ylabel("A(t) [Hz]")
    ax_rate_A_0.set_title("Activity Pop A", fontsize=10)

    evsB = smB["events"]["senders"]
    tsB = smB["events"]["times"]
    B_N_B = np.ones((t.size, 1)) * np.nan
    trmB = rmB["events"]["times"]
    trmB = trmB * dt - t0
    bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
    B_N_B = np.histogram(trmB, bins=bins)[0] / order*2 / dt_rec
    B_N_B = B_N_B*1000
    ax_rate_B_0.plot(t, B_N_B, color='blue', label ='pop B')
    ax_rate_B_0.vlines(start_stim, 0, 40, color='grey')
    ax_rate_B_0.vlines(end_stim, 0, 40, color='grey')
    ax_rate_B_0.set_ylabel("A(t) [Hz]")
    ax_rate_B_0.set_title("Activity Pop B", fontsize=10)

    activity = {'time':t,'activity (Hz) pop_A': A_N_A, 'activity (Hz) pop_B': B_N_B}
    frequency = pd.DataFrame(activity)    
    frequency.to_csv(saving_dir+notes+'coh_0-'+ str(coherence)[2:]+'_frequency.csv')

    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
    t0 = nest.GetKernelStatus('time')
    results, stimulus_A, stimulus_B, noise_A, noise_B = simulate_network(1,coherence, order , start_stim , end_stim , simtime)     
    
    smA = nest.GetStatus(results["spike_monitor_A"])[0]
    rmA = nest.GetStatus(results["rate_monitor_A"])[0]	
    smB = nest.GetStatus(results["spike_monitor_B"])[0]
    rmB = nest.GetStatus(results["rate_monitor_B"])[0] 

    evsA = smA["events"]["senders"]
    tsA = smA["events"]["times"]
    t = np.arange(0., simtime, dt_rec)
    A_N_A = np.ones((t.size, 1)) * np.nan
    trmA = rmA["events"]["times"]
    trmA = trmA * dt - t0
    bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
    A_N_A = np.histogram(trmA, bins=bins)[0] / order*2 / dt_rec
    A_N_A = A_N_A*1000
    ax_rate_A_0.plot(t, A_N_A, color='red', alpha= 0.7 ,label ='pop A')

    evsB = smB["events"]["senders"]
    tsB = smB["events"]["times"]
    B_N_B = np.ones((t.size, 1)) * np.nan
    trmB = rmB["events"]["times"]
    trmB = trmB * dt - t0
    bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
    B_N_B = np.histogram(trmB, bins=bins)[0] / order*2 / dt_rec
    B_N_B = B_N_B*1000
    ax_rate_B_0.plot(t, B_N_B, color='blue', alpha= 0.7 ,label ='pop B')
    
    activity = {'time':t,'activity (Hz) pop_A': A_N_A, 'activity (Hz) pop_B': B_N_B}
    frequency = pd.DataFrame(activity)    
    frequency.to_csv(saving_dir+'coh_0-'+ str(coherence)[2:]+'_frequency.csv') 

    coherence = 0.512

    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
    t0 = nest.GetKernelStatus('time')
    results, stimulus_A, stimulus_B, noise_A, noise_B = simulate_network(1,coherence, order , start_stim , end_stim , simtime, fn_tuned_par = "w_decrease.csv")     
    
    smA = nest.GetStatus(results["spike_monitor_A"])[0]
    rmA = nest.GetStatus(results["rate_monitor_A"])[0]	
    smB = nest.GetStatus(results["spike_monitor_B"])[0]
    rmB = nest.GetStatus(results["rate_monitor_B"])[0] 

    evsA = smA["events"]["senders"]
    tsA = smA["events"]["times"]
    t = np.arange(0., simtime, dt_rec)
    A_N_A = np.ones((t.size, 1)) * np.nan
    trmA = rmA["events"]["times"]
    trmA = trmA * dt - t0
    bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
    A_N_A = np.histogram(trmA, bins=bins)[0] / order*2 / dt_rec
    A_N_A = A_N_A*1000
    ax_rate_A_51.plot(t, A_N_A, color='red', label ='pop A')
    ax_rate_A_51.vlines(start_stim, 0, 40, color='grey')
    ax_rate_A_51.vlines(end_stim, 0, 40, color='grey')
    ax_rate_A_51.set_ylabel("A(t) [Hz]")
    ax_rate_A_51.set_title("Activity Pop A", fontsize=10)

    evsB = smB["events"]["senders"]
    tsB = smB["events"]["times"]
    B_N_B = np.ones((t.size, 1)) * np.nan
    trmB = rmB["events"]["times"]
    trmB = trmB * dt - t0
    bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
    B_N_B = np.histogram(trmB, bins=bins)[0] / order*2 / dt_rec
    B_N_B = B_N_B*1000
    ax_rate_B_51.plot(t, B_N_B, color='blue', label ='pop B')
    ax_rate_B_51.vlines(start_stim, 0, 40, color='grey')
    ax_rate_B_51.vlines(end_stim, 0, 40, color='grey')
    ax_rate_B_51.set_ylabel("A(t) [Hz]")
    ax_rate_B_51.set_title("Activity Pop B", fontsize=10)

    activity = {'time':t,'activity (Hz) pop_A': A_N_A, 'activity (Hz) pop_B': B_N_B}
    frequency = pd.DataFrame(activity)    
    frequency.to_csv(saving_dir+notes+'coh_0-'+ str(coherence)[2:]+'_frequency.csv')

    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
    t0 = nest.GetKernelStatus('time')
    results, stimulus_A, stimulus_B, noise_A, noise_B = simulate_network(1,coherence, order , start_stim , end_stim , simtime)     
    
    smA = nest.GetStatus(results["spike_monitor_A"])[0]
    rmA = nest.GetStatus(results["rate_monitor_A"])[0]	
    smB = nest.GetStatus(results["spike_monitor_B"])[0]
    rmB = nest.GetStatus(results["rate_monitor_B"])[0] 

    evsA = smA["events"]["senders"]
    tsA = smA["events"]["times"]
    t = np.arange(0., simtime, dt_rec)
    A_N_A = np.ones((t.size, 1)) * np.nan
    trmA = rmA["events"]["times"]
    trmA = trmA * dt - t0
    bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
    A_N_A = np.histogram(trmA, bins=bins)[0] / order*2 / dt_rec
    A_N_A = A_N_A*1000
    ax_rate_A_51.plot(t, A_N_A, color='red', alpha= 0.7 ,label ='pop A')
    ax_rate_A_51.set_ylabel("A(t) [Hz]")
    ax_rate_A_51.set_title("Activity Pop A", fontsize=10)

    evsB = smB["events"]["senders"]
    tsB = smB["events"]["times"]
    B_N_B = np.ones((t.size, 1)) * np.nan
    trmB = rmB["events"]["times"]
    trmB = trmB * dt - t0
    bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
    B_N_B = np.histogram(trmB, bins=bins)[0] / order*2 / dt_rec
    B_N_B = B_N_B*1000
    ax_rate_B_51.plot(t, B_N_B, color='blue', alpha= 0.7 ,label ='pop B')

    activity = {'time':t,'activity (Hz) pop_A': A_N_A, 'activity (Hz) pop_B': B_N_B}
    frequency = pd.DataFrame(activity)    
    frequency.to_csv(saving_dir+'coh_0-'+ str(coherence)[2:]+'_frequency.csv')

    fig_2a.savefig(saving_dir+'Figure2a.eps' , bbox_inches='tight')
    plt.close()

if figure_2b: 
    coherence = 0.064
    
if figure_2c: 
    fig_2c, ((ax_rate_A_0,ax_rate_B_0),(ax_rate_A_51,ax_rate_B_51)) = plt.subplots(2, 2, sharex=True,  figsize=(5,3))
    notes = 'decrease_weight'
    saving_dir = 'results/'+fig_n+'/'+notes+'/'
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    coherence = 0.0

    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
    t0 = nest.GetKernelStatus('time')
    results, stimulus_A, stimulus_B, noise_A, noise_B = simulate_network(1,coherence, order , start_stim , end_stim , simtime, fn_tuned_par = "no_NMDA.csv")     
    
    smA = nest.GetStatus(results["spike_monitor_A"])[0]
    rmA = nest.GetStatus(results["rate_monitor_A"])[0]	
    smB = nest.GetStatus(results["spike_monitor_B"])[0]
    rmB = nest.GetStatus(results["rate_monitor_B"])[0] 

    evsA = smA["events"]["senders"]
    tsA = smA["events"]["times"]
    t = np.arange(0., simtime, dt_rec)
    A_N_A = np.ones((t.size, 1)) * np.nan
    trmA = rmA["events"]["times"]
    trmA = trmA * dt - t0
    bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
    A_N_A = np.histogram(trmA, bins=bins)[0] / order*2 / dt_rec
    A_N_A = A_N_A*1000
    ax_rate_A_0.plot(t, A_N_A, color='red', label ='pop A')
    ax_rate_A_0.vlines(start_stim, 0, 40, color='grey')
    ax_rate_A_0.vlines(end_stim, 0, 40, color='grey')
    ax_rate_A_0.set_ylabel("A(t) [Hz]")
    ax_rate_A_0.set_title("Activity Pop A", fontsize=10)

    evsB = smB["events"]["senders"]
    tsB = smB["events"]["times"]
    B_N_B = np.ones((t.size, 1)) * np.nan
    trmB = rmB["events"]["times"]
    trmB = trmB * dt - t0
    bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
    B_N_B = np.histogram(trmB, bins=bins)[0] / order*2 / dt_rec
    B_N_B = B_N_B*1000
    ax_rate_B_0.plot(t, B_N_B, color='blue', label ='pop B')
    ax_rate_B_0.vlines(start_stim, 0, 40, color='grey')
    ax_rate_B_0.vlines(end_stim, 0, 40, color='grey')
    ax_rate_B_0.set_ylabel("A(t) [Hz]")
    ax_rate_B_0.set_title("Activity Pop B", fontsize=10)

    activity = {'time':t,'activity (Hz) pop_A': A_N_A, 'activity (Hz) pop_B': B_N_B}
    frequency = pd.DataFrame(activity)    
    frequency.to_csv(saving_dir+notes+'coh_0-'+ str(coherence)[2:]+'_frequency.csv')


    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
    t0 = nest.GetKernelStatus('time')
    results, stimulus_A, stimulus_B, noise_A, noise_B = simulate_network(1,coherence, order , start_stim , end_stim , simtime)     
    
    smA = nest.GetStatus(results["spike_monitor_A"])[0]
    rmA = nest.GetStatus(results["rate_monitor_A"])[0]	
    smB = nest.GetStatus(results["spike_monitor_B"])[0]
    rmB = nest.GetStatus(results["rate_monitor_B"])[0] 

    evsA = smA["events"]["senders"]
    tsA = smA["events"]["times"]
    t = np.arange(0., simtime, dt_rec)
    A_N_A = np.ones((t.size, 1)) * np.nan
    trmA = rmA["events"]["times"]
    trmA = trmA * dt - t0
    bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
    A_N_A = np.histogram(trmA, bins=bins)[0] / order*2 / dt_rec
    A_N_A = A_N_A*1000
    ax_rate_A_0.plot(t, A_N_A, color='red', alpha= 0.7 ,label ='pop A')

    evsB = smB["events"]["senders"]
    tsB = smB["events"]["times"]
    B_N_B = np.ones((t.size, 1)) * np.nan
    trmB = rmB["events"]["times"]
    trmB = trmB * dt - t0
    bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
    B_N_B = np.histogram(trmB, bins=bins)[0] / order*2 / dt_rec
    B_N_B = B_N_B*1000
    ax_rate_B_0.plot(t, B_N_B, color='blue', alpha= 0.7 ,label ='pop B') 

    activity = {'time':t,'activity (Hz) pop_A': A_N_A, 'activity (Hz) pop_B': B_N_B}
    frequency = pd.DataFrame(activity)    
    frequency.to_csv(saving_dir+'coh_0-'+ str(coherence)[2:]+'_frequency_2.csv')

    coherence = 0.512

    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
    t0 = nest.GetKernelStatus('time')
    results, stimulus_A, stimulus_B, noise_A, noise_B = simulate_network(1,coherence, order , start_stim , end_stim , simtime, fn_tuned_par = "no_NMDA.csv")     
    
    smA = nest.GetStatus(results["spike_monitor_A"])[0]
    rmA = nest.GetStatus(results["rate_monitor_A"])[0]	
    smB = nest.GetStatus(results["spike_monitor_B"])[0]
    rmB = nest.GetStatus(results["rate_monitor_B"])[0] 

    evsA = smA["events"]["senders"]
    tsA = smA["events"]["times"]
    t = np.arange(0., simtime, dt_rec)
    A_N_A = np.ones((t.size, 1)) * np.nan
    trmA = rmA["events"]["times"]
    trmA = trmA * dt - t0
    bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
    A_N_A = np.histogram(trmA, bins=bins)[0] / order*2 / dt_rec
    A_N_A = A_N_A*1000
    ax_rate_A_51.plot(t, A_N_A, color='red', label ='pop A')
    ax_rate_A_51.vlines(start_stim, 0, 40, color='grey')
    ax_rate_A_51.vlines(end_stim, 0, 40, color='grey')
    ax_rate_A_51.set_ylabel("A(t) [Hz]")
    ax_rate_A_51.set_title("Activity Pop A", fontsize=10)

    evsB = smB["events"]["senders"]
    tsB = smB["events"]["times"]
    B_N_B = np.ones((t.size, 1)) * np.nan
    trmB = rmB["events"]["times"]
    trmB = trmB * dt - t0
    bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
    B_N_B = np.histogram(trmB, bins=bins)[0] / order*2 / dt_rec
    B_N_B = B_N_B*1000
    ax_rate_B_51.plot(t, B_N_B, color='blue', label ='pop B')
    ax_rate_B_51.vlines(start_stim, 0, 40, color='grey')
    ax_rate_B_51.vlines(end_stim, 0, 40, color='grey')
    ax_rate_B_51.set_ylabel("A(t) [Hz]")
    ax_rate_B_51.set_title("Activity Pop B", fontsize=10)

    activity = {'time':t,'activity (Hz) pop_A': A_N_A, 'activity (Hz) pop_B': B_N_B}
    frequency = pd.DataFrame(activity)    
    frequency.to_csv(saving_dir+notes+'coh_0-'+ str(coherence)[2:]+'_frequency.csv')

    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
    t0 = nest.GetKernelStatus('time')
    results, stimulus_A, stimulus_B, noise_A, noise_B = simulate_network(1,coherence, order , start_stim , end_stim , simtime)     
    
    smA = nest.GetStatus(results["spike_monitor_A"])[0]
    rmA = nest.GetStatus(results["rate_monitor_A"])[0]	
    smB = nest.GetStatus(results["spike_monitor_B"])[0]
    rmB = nest.GetStatus(results["rate_monitor_B"])[0] 

    evsA = smA["events"]["senders"]
    tsA = smA["events"]["times"]
    t = np.arange(0., simtime, dt_rec)
    A_N_A = np.ones((t.size, 1)) * np.nan
    trmA = rmA["events"]["times"]
    trmA = trmA * dt - t0
    bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
    A_N_A = np.histogram(trmA, bins=bins)[0] / order*2 / dt_rec
    A_N_A = A_N_A*1000
    ax_rate_A_51.plot(t, A_N_A, color='red', alpha= 0.7 ,label ='pop A')
    ax_rate_A_51.set_ylabel("A(t) [Hz]")
    ax_rate_A_51.set_title("Activity Pop A", fontsize=10)

    evsB = smB["events"]["senders"]
    tsB = smB["events"]["times"]
    B_N_B = np.ones((t.size, 1)) * np.nan
    trmB = rmB["events"]["times"]
    trmB = trmB * dt - t0
    bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
    B_N_B = np.histogram(trmB, bins=bins)[0] / order*2 / dt_rec
    B_N_B = B_N_B*1000
    ax_rate_B_51.plot(t, B_N_B, color='blue', alpha= 0.7 ,label ='pop B')

    activity = {'time':t,'activity (Hz) pop_A': A_N_A, 'activity (Hz) pop_B': B_N_B}
    frequency = pd.DataFrame(activity)    
    frequency.to_csv(saving_dir+'coh_0-'+ str(coherence)[2:]+'_frequency_2.csv')
    
    notes = 'no_NMDA'
    saving_dir = 'results/'+fig_n+'/'+notes+'/'
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    fig_2c.savefig(saving_dir+'Figure2c.eps' , bbox_inches='tight')
    plt.close()
