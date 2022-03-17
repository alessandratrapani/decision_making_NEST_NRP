import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nest
import nest.raster_plot
import os
from simulate_network import *

#sim 1 c51.2 1 c12.8, 1 0.0, save also inhi activity, average 100 sim for decison space plot
order = 400
simtime = 3000.0
start_stim = 200.0
end_stim = 1200.0
dt = 0.1
dt_rec = 10.0
rec_pop = 0.5

save = False

figure1abcd = True
figure1ed = True

fig_n = 'Figure1'

if figure1abcd:
    fig_1d, ax_rate_in = plt.subplots(1, 1,  figsize=(3,3))
    mult_coherence = [0.0, 0.128, 0.512]
    for i,coherence in enumerate(mult_coherence):
        fig_1abc, ((ax_raster_A,ax_raster_B), (ax_rate_A,ax_rate_B)) = plt.subplots(2, 2, sharex=True,  figsize=(5,3))
        nest.ResetKernel()
        nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
        t0 = nest.GetKernelStatus('time')

        results, stimulus_A, stimulus_B, noise_A, noise_B = simulate_network(1,coherence, order , start_stim , end_stim , simtime, rec_pop = rec_pop)     

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
        A_N_A = np.histogram(trmA, bins=bins)[0] / order*2*rec_pop / dt_rec
        A_N_A = A_N_A*1000

        evsB = smB["events"]["senders"]
        tsB = smB["events"]["times"]
        B_N_B = np.ones((t.size, 1)) * np.nan
        trmB = rmB["events"]["times"]
        trmB = trmB * dt - t0
        bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
        B_N_B = np.histogram(trmB, bins=bins)[0] / order*2*rec_pop / dt_rec
        B_N_B = B_N_B*1000

        evsIn = smIn["events"]["senders"]
        tsIn = smIn["events"]["times"]
        I_N_I = np.ones((t.size, 1)) * np.nan
        trmIn = rmIn["events"]["times"]
        trmIn = trmIn * dt - t0
        bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
        I_N_I = np.histogram(trmIn, bins=bins)[0] / order*1*rec_pop / dt_rec
        I_N_I = I_N_I*1000

        ax_rate_in.plot(t, I_N_I, color='black',alpha=1-i*0.1, label ='coh_' + '0-'+ str(coherence)[2:])

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
            notes = 'coh_' + '0-'+ str(coherence)[2:] + '_trial_single'
            saving_dir = 'results/'+fig_n+'/'+notes+'/'
            if not os.path.exists(saving_dir):
                os.makedirs(saving_dir)

            raster_A = {'ID neuron pop_A':evsA, 'event time pop_A':tsA}
            raster_B = { 'ID neuron pop_B':evsB, 'event time pop_B':tsB}
            raster_In = { 'ID neuron pop_inh':evsIn, 'event time pop_inh':tsIn}
            activity = {'time':t,'activity (Hz) pop_A': A_N_A, 'activity (Hz) pop_B': B_N_B, 'activity (Hz) pop_inh': I_N_I}
            
            events_A = pd.DataFrame(raster_A)
            events_B = pd.DataFrame(raster_B)
            events_inh = pd.DataFrame(raster_In)
            frequency = pd.DataFrame(activity)
            
            events_A.to_csv(saving_dir+notes+'_events_pop_A.csv')
            events_B.to_csv(saving_dir+notes+'_events_pop_B.csv')
            events_inh.to_csv(saving_dir+notes+'_events_pop_inh.csv')
            frequency.to_csv(saving_dir+notes+'_frequency.csv')

            fig_1abc.savefig(saving_dir+'Figure1abc.eps' , bbox_inches='tight')
            plt.close()

        else:
            plt.show() 

    ax_rate_in.set_ylabel("A(t) [Hz]")
    ax_rate_in.set_title("Activity Pop Inh", fontsize=10)
    ax_rate_in.legend()
    fig_1d.savefig('results/'+fig_n+'/Figure1d.eps', bbox_inches='tight')

if figure1ed:
    fig_1d, ax_rate_in = plt.subplots(1, 1,  figsize=(3,3))
    mult_coherence = [0.0, 0.128, 0.512]
    n_trial = 10
    winner = np.zeros((len(mult_coherence),2))
    for i,coherence in enumerate(mult_coherence):
        win_A=0
        win_B=0
        I_N_I_mean = 0

        for j in range(n_trial):  
        
            nest.ResetKernel()
            nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
            t0 = nest.GetKernelStatus('time')

            results, stimulus_A, stimulus_B, noise_A, noise_B = simulate_network(j,coherence, order , start_stim , end_stim , simtime)     

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
            A_N_A = np.histogram(trmA, bins=bins)[0] / order*2*rec_pop / dt_rec
            A_N_A = A_N_A*1000

            evsB = smB["events"]["senders"]
            tsB = smB["events"]["times"]
            B_N_B = np.ones((t.size, 1)) * np.nan
            trmB = rmB["events"]["times"]
            trmB = trmB * dt - t0
            bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
            B_N_B = np.histogram(trmB, bins=bins)[0] / order*2*rec_pop / dt_rec
            B_N_B = B_N_B*1000

            evsIn = smIn["events"]["senders"]
            tsIn = smIn["events"]["times"]
            I_N_I = np.ones((t.size, 1)) * np.nan
            trmIn = rmIn["events"]["times"]
            trmIn = trmIn * dt - t0
            bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
            I_N_I = np.histogram(trmIn, bins=bins)[0] / order*1*rec_pop / dt_rec
            I_N_I = I_N_I*1000

            I_N_I_mean = I_N_I_mean + I_N_I/n_trial

            if np.mean(A_N_A[-100:-1])>np.mean(B_N_B[-100:-1]):
                win_pop = '_winA'
                win_A = win_A + 1
                winner[i,0]=win_A
                c = 'red'
            elif np.mean(A_N_A[-100:-1])<np.mean(B_N_B[-100:-1]):
                win_pop = '_winB'
                win_B = win_B + 1
                winner[i,1]=win_B
                c = 'blue'

            if save:
                notes = 'coh_' + '0-'+ str(coherence)[2:] + '_trial_'+ str(j)
                saving_dir = 'results/'+fig_n+'/'+notes+'/'
                if not os.path.exists(saving_dir):
                    os.makedirs(saving_dir)

                raster_A = {'ID neuron pop_A':evsA, 'event time pop_A':tsA}
                raster_B = { 'ID neuron pop_B':evsB, 'event time pop_B':tsB}
                raster_In = { 'ID neuron pop_inh':evsIn, 'event time pop_inh':tsIn}
                activity = {'time':t,'activity (Hz) pop_A': A_N_A, 'activity (Hz) pop_B': B_N_B, 'activity (Hz) pop_inh': I_N_I}
                
                events_A = pd.DataFrame(raster_A)
                events_B = pd.DataFrame(raster_B)
                events_inh = pd.DataFrame(raster_In)
                frequency = pd.DataFrame(activity)
                
                events_A.to_csv(saving_dir+notes+'_events_pop_A'+win_pop+'.csv')
                events_B.to_csv(saving_dir+notes+'_events_pop_B'+win_pop+'.csv')
                events_inh.to_csv(saving_dir+notes+'_events_pop_inh'+win_pop+'.csv')
                frequency.to_csv(saving_dir+notes+'_frequency'+win_pop+'.csv')
        
        ax_rate_in.plot(t, I_N_I_mean, color='black',alpha=1-i*0.1, label ='coh_' + '0-'+ str(coherence)[2:])

    ax_rate_in.set_ylabel("A(t) [Hz]")
    ax_rate_in.set_title("Activity Pop Inh", fontsize=10)
    ax_rate_in.legend()
    fig_1d.savefig('results/'+fig_n+'/Figure1d.eps', bbox_inches='tight')
    
    win = {'coherence': mult_coherence, 'pop A win': winner[:,0], 'pop B win': winner[:,1]}
    win = pd.DataFrame(win)
    win.to_csv('results/'+fig_n+'/'+fig_n+'_winners.csv')
    sim_info = {'n_trial':n_trial, 'sim time':simtime, 'start sim': start_stim, 'end sim': end_stim, 'order':order, 'dt_rec':dt_rec}
    sim_info = pd.DataFrame(sim_info, index = ['value'])
    sim_info.to_csv('results/'+fig_n+'/'+fig_n+'_sim_info.csv')

    fig_1e, dec_space = plt.subplots(1, 1,  figsize=(3,3))
    mult_coherence = [0.0, 0.128, 0.512]
    n_trial = 100
    winner = np.zeros((len(mult_coherence),2))
    win_pop = 'pop_B'
    for i,coherence in enumerate(mult_coherence):
        A_N_A_mean = np.array([])
        B_N_B_mean = np.array([])
        for j in range(n_trial):  
            notes = 'coh_' + '0-'+ str(coherence)[2:] + '_trial_'+ str(j)
            file_name = 'results/'+fig_n+'/'+notes+'/'+notes+'_frequency'+win_pop+'.csv'
            if os.path.exists(file_name):
                activity = pd.read_csv(file_name)
                A_N_A = activity['activity (Hz) pop_A'].to_numpy()
                A_N_A_mean = np.append(A_N_A_mean,A_N_A)
                B_N_B = activity['activity (Hz) pop_B'].to_numpy()
                B_N_B_mean = np.append(B_N_B_mean,B_N_B)

        A_N_A_mean = np.mean(A_N_A_mean)
        B_N_B_mean = np.mean(B_N_B_mean)

        dec_space.plot(A_N_A,B_N_B, color='blue', alpha=1-i*0.1, label ='coh_' + '0-'+ str(coherence)[2:])

    dec_space.set_xlim(-0.1,40)
    dec_space.set_ylim(-0.1,40)
    dec_space.set_xlabel("Firing rate pop A (Hz)")
    dec_space.set_ylabel("Firing rate pop B (Hz)")
    dec_space.set_title("Decision Space")
    dec_space.legend()
    fig_1e.savefig('results/'+fig_n+'/Figure1e.eps', bbox_inches='tight')