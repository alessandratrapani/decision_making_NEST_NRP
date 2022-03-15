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
# @coherence= 0.0    --> A B (da wang circa 50%)
# @coherence= 0.032  --> A B (da wang circa 55%)
# @coherence= 0.064  --> A B (da wang circa 70%)
# @coherence= 0.128  --> A B (da wang circa 90%)
# @coherence= 0.256  --> A B (da wang circa 100%)
# @coherence= 0.512  --> A B (da wang circa 100%)
# @coherence= 1.     --> A B (da wang100%)
# @coherence= -0.032 --> A B (da wang circa 55%)
# @coherence= -0.064 --> A B (da wang circa 70%)
# @coherence= -0.128 --> A B (da wang circa 90%)
# @coherence= -0.256 --> A B (da wang circa 100%)
# @coherence= -0.512 --> A B (da wang circa 100%)
# @coherence= -1.    --> A B (da wang100%)

coherence = -0.128
order = 400
simtime = 3000.0
start_stim = 500.0
end_stim = 1500.0

show_fig = True
save = True
#mult_coherence = [0.0, 0.032, 0.064, 0.128, 0.256, 0.512, 1., -0.032, -0.064, -0.128, -0.256, -0.512, -1.]
mult_coherence= [0.0, 0.128, 0.512]
n_trial = 3
winner = np.zeros((len(mult_coherence),2))

for i,coherence in enumerate(mult_coherence):
    win_A=0
    win_B=0
    delta_s_A_winner = []
    delta_s_B_winner = []
    for j in range(n_trial):    
        nest.ResetKernel()
        dt = 0.1
        dt_rec = 10.0
        nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
        t0 = nest.GetKernelStatus('time')

        results, stimulus_A, stimulus_B, noise_A, noise_B, sum_stimulus_A, sum_stimulus_B = simulate_network(j,coherence, order , start_stim , end_stim , simtime)     

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

        evsB = smB["events"]["senders"]
        tsB = smB["events"]["times"]
        B_N_B = np.ones((t.size, 1)) * np.nan
        trmB = rmB["events"]["times"]
        trmB = trmB * dt - t0
        bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
        B_N_B = np.histogram(trmB, bins=bins)[0] / order*2 / dt_rec
        B_N_B = B_N_B*1000

        if np.mean(A_N_A[-10:-1])>np.mean(B_N_B[-10:-1]):
            win_A = win_A + 1
            winner[i,0]=win_A
            c = 'red'
            #TODO check
            delta_s_A_winner.append(sum_stimulus_A[-1] - sum_stimulus_B[-1])
            print('pop_A ', np.mean(A_N_A[-10:-1]))
            print('pop_B ', np.mean(B_N_B[-10:-1]))
        else:
            win_B = win_B + 1
            winner[i,1]=win_B
            c = 'blue'
            delta_s_B_winner.append(sum_stimulus_A[-1] - sum_stimulus_B[-1])
            print('pop_A ', np.mean(A_N_A[-10:-1]))
            print('pop_B ', np.mean(B_N_B[-10:-1]))

        
        
        fig = None
        ax_raster = None
        ax_rate = None
        fig, (ax_raster, ax_rate, ax_stimuli,ax_noise) = plt.subplots(4, 1, sharex=True, figsize=(16,9))
        plt.suptitle('Coherence ' + str(coherence*100) + '%')
        
        ax_raster.plot(tsA, evsA, ".", color='red', label ='pop A')
        ax_raster.plot(tsB, evsB, ".", color='blue', label ='pop B')
        ax_raster.set_ylabel("neuron #")
        ax_raster.set_title("Raster Plot ", fontsize=10)
        ax_raster.legend()
        ax_rate.plot(t, A_N_A, color='red', label ='pop A')
        ax_rate.fill_between(t, A_N_A, color='red')
        ax_rate.plot(t, B_N_B, color='blue', label ='pop B')
        ax_rate.fill_between(t, B_N_B, color='blue')
        ax_rate.vlines(start_stim, 0, 40, color='grey')
        ax_rate.vlines(end_stim, 0, 40, color='grey')
        ax_rate.set_ylabel("A(t) [Hz]")
        ax_rate.set_title("Activity", fontsize=10)
        ax_rate.legend()
        ax_stimuli.plot(np.arange(0., simtime),stimulus_A, 'red', label='stimulus on A')
        ax_stimuli.plot(np.arange(0., simtime),stimulus_B, 'blue', label='stimulus on B')
        ax_stimuli.legend()
        ax_noise.plot(np.arange(0., simtime),noise_A, 'orange', label='noise on A')
        ax_noise.plot(np.arange(0., simtime),noise_B, 'lightblue', label='noise on B')
        ax_noise.legend()            
        plt.xlabel("t [ms]")
        

        # decisional_space = plt.figure(figsize = [10,10])                
        # plt.plot(A_N_A, B_N_B, color=c)
        # plt.plot([0,40],[0,40], color='grey')
        # plt.xlim(-0.1,40)
        # plt.ylim(-0.1,40)
        # plt.xlabel("Firing rate pop A (Hz)")
        # plt.ylabel("Firing rate pop B (Hz)")
        # plt.title("Decision Space")
        if show_fig:
            plt.show()

        if save:
            notes = 'coh_' + '0-'+ str(coherence)[2:] + '_trial_'+ str(j)
            saving_dir = 'results/'+dt_string+'/'+notes+'/'
            if not os.path.exists(saving_dir):
                os.makedirs(saving_dir)

            raster_A = {'ID neuron pop_A':evsA, 'event time pop_A':tsA}
            raster_B = { 'ID neuron pop_B':evsB, 'event time pop_B':tsB}
            activity = {'time':t,'activity (Hz) pop_A': A_N_A, 'activity (Hz) pop_B': B_N_B}
            stimuli = {'stimulus pop A': stimulus_A,'stimulus pop B': stimulus_B, 'integral stim pop A': sum_stimulus_A,'integral stim pop B': sum_stimulus_B}
            
            events_A = pd.DataFrame(raster_A)
            events_B = pd.DataFrame(raster_B)
            frequency = pd.DataFrame(activity)
            stimuli = pd.DataFrame(stimuli)
            
            events_A.to_csv(saving_dir+notes+'_events_pop_A.csv')
            events_B.to_csv(saving_dir+notes+'_events_pop_B.csv')
            frequency.to_csv(saving_dir+notes+'_frequency.csv')
            stimuli.to_csv(saving_dir+notes+'_stimuli.csv')


win = {'coherence': mult_coherence, 'pop A win': winner[:,0], 'pop B win': winner[:,1]}
win = pd.DataFrame(win)
win.to_csv('results/'+dt_string+'/'+dt_string+'_winners.csv')
sim_info = {'n_trial':n_trial, 'sim time':simtime, 'start sim': start_stim, 'end sim': end_stim, 'order':order}
sim_info = pd.DataFrame(sim_info)
sim_info.to_csv('results/'+dt_string+'/'+dt_string+'_sim_info.csv')

print("win_A", win_A)
print("win_B", win_B)
print(winner)
print(dt_string)