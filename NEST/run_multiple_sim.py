import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nest
import nest.raster_plot
import os
from simulate_network import *

# @coherence= 0.0    --> 7A 3B (da wang circa 50%)
# @coherence= 0.032  --> A B (da wang circa 55%)
# @coherence= 0.064  --> A B (da wang circa 70%)
# @coherence= 0.128  --> 3A 7B (da wang circa 90%)
# @coherence= 0.256  --> 2A 8B (da wang circa 100%)
# @coherence= 0.512  --> A B (da wang circa 100%)
# @coherence= 1.     --> A B (da wang100%)
# @coherence= -0.032 --> 8A 2B (da wang circa 55%)
# @coherence= -0.064 --> 7A 3B (da wang circa 70%)
# @coherence= -0.128 --> 8A 2B (da wang circa 90%)
# @coherence= -0.256 --> 10A 0B (da wang circa 100%)
# @coherence= -0.512 --> A B (da wang circa 100%)
# @coherence= -1.    --> A B (da wang100%)

coherence = -0.128
order = 400
simtime = 3000.0
start_stim = 500.0
end_stim = 1000.0

show_fig = False

#mult_coherence = [0.0, 0.032, 0.064, 0.128, 0.256, 0.512, 1., -0.032, -0.064, -0.128, -0.256, -0.512, -1.]
mult_coherence= [0.128,-0.128]
winner = np.zeros((len(mult_coherence),2))

for i,coherence in enumerate(mult_coherence):
    win_A=0
    win_B=0
    for j in range(10):    
        nest.ResetKernel()
        dt = 0.1
        dt_rec = 10.0
        nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
        t0 = nest.GetKernelStatus('time')

        results,stimulus_A, stimulus_B, noise_A, noise_B = simulate_network(j,coherence, order , start_stim , end_stim , simtime)     

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
        A_N_A = np.histogram(trmA, bins=bins)[0] / order / dt_rec

        evsB = smB["events"]["senders"]
        tsB = smB["events"]["times"]
        B_N_B = np.ones((t.size, 1)) * np.nan
        trmB = rmB["events"]["times"]
        trmB = trmB * dt - t0
        bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
        B_N_B = np.histogram(trmB, bins=bins)[0] / order / dt_rec

        if np.mean(A_N_A[-10:-1]*1000)>np.mean(B_N_B[-10:-1]*1000):
            win_A = win_A + 1
            winner[i,0]=win_A
            print('pop_A ', np.mean(A_N_A[-10:-1]*1000))
            print('pop_B ', np.mean(B_N_B[-10:-1]*1000))
        else:
            win_B = win_B + 1
            winner[i,1]=win_B
            print('pop_A ', np.mean(A_N_A[-10:-1]*1000))
            print('pop_B ', np.mean(B_N_B[-10:-1]*1000))
        
        if show_fig:
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
            ax_rate.plot(t, A_N_A * 1000, color='red', label ='pop A')
            ax_rate.fill_between(t, A_N_A * 1000, color='red')
            ax_rate.plot(t, B_N_B * 1000, color='blue', label ='pop B')
            ax_rate.fill_between(t, B_N_B * 1000, color='blue')
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
            plt.show()

print("win_A", win_A)
print("win_B", win_B)
print(winner)