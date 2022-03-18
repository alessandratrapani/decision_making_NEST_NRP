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
save = False

if not os.path.exists('figures/'+fig_n+'/'):
    os.makedirs('figures/'+fig_n+'/')

dt = 0.1
dt_rec = 10.0
n_trial = 1
start_stim = 200.0
simtime = 2500.0
order = 400
stimulus_update_interval = 25
fn_fixed_par = "fixed_parameters.csv"
fn_tuned_par = "tuned_parameters.csv"
rec_pop=1.

figure7a = False
figure7b = False
figure7c = True

if run_t_reverse:
    coherence = 0.128
    stim_rev = -0.8
    multiple_t_rev = [100+start_stim, 300+start_stim, 500+start_stim, 700+start_stim, 800+start_stim]
    end_stim = 1200.0
    winner = np.zeros((len(multiple_t_rev),2))
    results_dir = 'results/t_reverse/'
    for i,t_rev in enumerate(multiple_t_rev):
        win_A=0
        win_B=0
        delta_s_A_winner = []
        delta_s_B_winner = []
        for j in range(n_trial):    
            nest.ResetKernel()
            nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
            t0 = nest.GetKernelStatus('time')

            results, stimulus_A, stimulus_B, noise_A, noise_B = simulate_network_revstim(t_rev=t_rev, stim_rev=stim_rev, n_run=1,coherence = coherence, order = 400, start_stim = 500.0, end_stim = 1000.0, simtime = 3000.0, stimulus_update_interval = 25, fn_fixed_par = "fixed_parameters.csv", fn_tuned_par = "tuned_parameters.csv", rec_pop=1.)
    
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

            for n in range(1,int(simtime)):
                int_stimulus_A[n] = int_stimulus_A[n-1]+stimulus_A[n]
                int_stimulus_B[n] = int_stimulus_B[n-1]+stimulus_B[n]

            if np.mean(A_N_A[-100:-1])>np.mean(B_N_B[-100:-1]):
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
            stimuli = {'stimulus pop A': stimulus_A,'stimulus pop B': stimulus_B}#, 'integral stim pop A': int_stimulus_A,'integral stim pop B': int_stimulus_B}
            
            events_A = pd.DataFrame(raster_A)
            events_B = pd.DataFrame(raster_B)
            events_inh = pd.DataFrame(raster_In)
            frequency = pd.DataFrame(activity)
            stimuli = pd.DataFrame(stimuli)
            
            events_A.to_csv(saving_dir+'events_pop_A.csv')
            events_B.to_csv(saving_dir+'events_pop_B.csv')
            frequency.to_csv(saving_dir+'frequency.csv')
            events_inh.to_csv(saving_dir+'events_pop_inh.csv')
            stimuli.to_csv(saving_dir+'stimuli.csv')

        delta_s_A_winner = {'delta_s_A_winner':delta_s_A_winner}
        delta_s_B_winner = {'delta_s_B_winner':delta_s_B_winner}
        delta_s_A_winner = pd.DataFrame(delta_s_A_winner)
        delta_s_B_winner = pd.DataFrame(delta_s_B_winner)
        delta_s_A_winner.to_csv(results_dir+'/t_rev'+str(t_rev) +'/delta_s_A_winner.csv')
        delta_s_B_winner.to_csv(results_dir+'/t_rev'+str(t_rev) +'/delta_s_B_winner.csv')

    win = {'t_rev': multiple_t_rev, 'pop A win': winner[:,0], 'pop B win': winner[:,1]}
    win = pd.DataFrame(win)
    win.to_csv(results_dir+'winners.csv')
    sim_info = {'n_trial':n_trial, 'sim time':simtime, 'start sim': start_stim, 'end sim': end_stim, 'order':order, 'dt_rec':dt_rec}
    sim_info = pd.DataFrame(sim_info, index = ['value'])
    sim_info.to_csv(results_dir+'sim_info.csv')

if run_stim_reverse:
    coherence = 0.128
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

            results, stimulus_A, stimulus_B, noise_A, noise_B = simulate_network_revstim(t_rev, stim_rev ,j,coherence, order , start_stim , end_stim , simtime,stimulus_update_interval, fn_fixed_par, fn_tuned_par, rec_pop)     

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


            if np.mean(A_N_A[-100:-1])>np.mean(B_N_B[-100:-1]):
                win_A = win_A + 1
                winner[i,0]=win_A
                c = 'red'
                win_pop = 'A_win'
            else:
                win_B = win_B + 1
                winner[i,1]=win_B
                c = 'blue'
                win_pop = 'B_win'

            
            notes = 'stim_rev'+ str(stim_rev)+'/'+win_pop+ '/trial_'+ str(j)
            saving_dir = results_dir+notes+'/'
            if not os.path.exists(saving_dir):
                os.makedirs(saving_dir)

            raster_A = {'ID neuron pop_A':evsA, 'event time pop_A':tsA}
            raster_B = { 'ID neuron pop_B':evsB, 'event time pop_B':tsB}
            raster_In = { 'ID neuron pop_inh':evsIn, 'event time pop_inh':tsIn}
            activity = {'time':t,'activity (Hz) pop_A': A_N_A, 'activity (Hz) pop_B': B_N_B, 'activity (Hz) pop_inh': I_N_I}
            stimuli = {'stimulus pop A': stimulus_A,'stimulus pop B': stimulus_B}#, 'integral stim pop A': int_stimulus_A,'integral stim pop B': int_stimulus_B}
            
            events_A = pd.DataFrame(raster_A)
            events_B = pd.DataFrame(raster_B)
            events_inh = pd.DataFrame(raster_In)
            frequency = pd.DataFrame(activity)
            stimuli = pd.DataFrame(stimuli)
            
            events_A.to_csv(saving_dir+'events_pop_A.csv')
            events_B.to_csv(saving_dir+'events_pop_B.csv')
            frequency.to_csv(saving_dir+'frequency.csv')
            events_inh.to_csv(saving_dir+'events_pop_inh.csv')
            stimuli.to_csv(saving_dir+'stimuli.csv')

    win = {'sti_rev': multiple_stim_rev, 'pop A win': winner[:,0], 'pop B win': winner[:,1]}
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
    n_trial=1
    
    coherence_level = winner['t_rev'].to_numpy()
    pop_A_win = 100*(winner['pop A win'].to_numpy())/n_trial
    pop_B_win = 100*(winner['pop B win'].to_numpy())/n_trial

    fig, ax1 = plt.subplots(1,1,figsize = [5,5])
    ax1.plot(coherence_level,pop_A_win,'*', color='red')
    ax1.plot(coherence_level,pop_B_win,'*', color='blue')
    ax1.set_xlabel('Coherence level %')
    ax1.set_ylabel('%\ of correct choice')
    if save:
        fig.savefig('figures/'+fig_n+'/Figure7A.eps' , bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if figure7b:
    save=False
    dt_string = 'prova'
    results_dir = 'results/stim_reverse/'
    winner = pd.read_csv(results_dir+'winners.csv')
    
    coherence_level = winner['sti_rev'].to_numpy()*100
    pop_A_win = 100*(winner['pop A win'].to_numpy())/n_trial
    pop_B_win = 100*(winner['pop B win'].to_numpy())/n_trial

    fig, ax1 = plt.subplots(1,1,figsize = [5,5])
    ax1.plot(coherence_level,pop_A_win,'*', color='red')
    ax1.plot(coherence_level,pop_B_win,'*', color='blue')
    ax1.set_xlabel('Coherence level %')
    ax1.set_ylabel('%\ of correct choice')
    if save:
        fig.savefig('figures/'+fig_n+'/Figure7B.eps' , bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if figure7c:
    fig,axes = plt.subplots(2,2,figsize = [16,10])
    stim_rev=-0.8
    dt_string = 'standard/'
    coherence = 0.128
    B_N_B_mean = []
    A_N_A_mean = []
    for j in range(n_trial): 
        #Solid curve black: time course of pop B winning with stimulus on B
        win_pop ='B_win'
        path = 'results/'+dt_string+'c'+str(coherence) +'/'+win_pop+ '/trial_'+ str(j)+'/'
        if os.path.exists(path):
            evsB, tsB, t, B_N_B, stimulus_B, sum_stimulus_B = extract_results(path, 'B')
            evsA, tsA, t, A_N_A, stimulus_A, sum_stimulus_A = extract_results(path, 'A')
            #axes[0].plot(t,B_N_B,'black')
            B_N_B_mean.append(B_N_B)   
            A_N_A_mean.append(A_N_A)    
    B_N_B_mean = np.mean(B_N_B_mean,axis=0)
    A_N_A_mean = np.mean(A_N_A_mean,axis=0)
    axes[0][0].plot(t,B_N_B_mean,'blue')
    axes[0][0].plot(t,A_N_A_mean,'red')
    axes[1][0].plot(np.arange(0., simtime),stimulus_B,'blue')
    axes[1][0].plot(np.arange(0., simtime),stimulus_A,'red')

    multiple_t_rev = [100+start_stim, 300+start_stim, 500+start_stim, 700+start_stim, 800+start_stim]
        
    for j in range(n_trial): 
        #Solid curve black: time course of pop B winning with stimulus on B
        win_pop ='A_win'
        path = 'results/stim_reverse/stim_rev'+ str(stim_rev)+'/'+win_pop+ '/trial_'+ str(j)+'/'
        B_N_B_mean = []
        A_N_A_mean = []
        if os.path.exists(path):
            frequency = pd.read_csv(path+'frequency.csv')
            stimuli = pd.read_csv(path+'stimuli.csv')
            t = frequency['time'].to_numpy()
            B_N_B = frequency['activity (Hz) pop_B'].to_numpy()
            A_N_A = frequency['activity (Hz) pop_A'].to_numpy()
            stimulus_A = stimuli['stimulus pop A'].to_numpy()
            stimulus_B = stimuli['stimulus pop B'].to_numpy()
            B_N_B_mean.append(B_N_B)   
            A_N_A_mean.append(A_N_A)    
    B_N_B_mean = np.mean(B_N_B_mean,axis=0)
    A_N_A_mean = np.mean(A_N_A_mean,axis=0)
    axes[0][1].plot(t,B_N_B_mean,'blue')
    axes[0][1].plot(t,A_N_A_mean,'red')
    axes[1][1].plot(np.arange(0., simtime),stimulus_B,'blue')
    axes[1][1].plot(np.arange(0., simtime),stimulus_A,'red')
    if save:
        fig.savefig('figures/'+fig_n+'/Figure7C.eps' , bbox_inches='tight')
        plt.close()
    else:
        plt.show()