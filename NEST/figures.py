import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#%% FIGURE 1
#TODO 1 0%, 12.8%, 51.2% raster and activity (Hz)
dt_string = 

results_directory = os.getcwd()+'/results/'+dt_string+'/'
sim_info = pd.read_csv(results_directory+dt_string+'_sim_info.csv')
simtime = sim_info['sim time'].to_numpy
start_stim = sim_info['start sim'].to_numpy
end_stim = sim_info['end sim'].to_numpy
order = sim_info['order'].to_numpy
sim_info = pd.DataFrame(sim_info)

coherence = 0.0
trial = 1

notes = 'coh_' + '0-'+ str(coherence)[2:] + '_trial_'+ str(trial)
trial_directory = results_directory + notes +'/'
events_A = pd.read_csv(trial_directory+notes+'_events_pop_A.csv')
evsA = events_A['ID neuron pop_A'].to_numpy
tsA = events_A['event time pop_A'].to_numpy
events_B = pd.read_csv(trial_directory+notes+'_events_pop_B.csv')
evsB = events_B['ID neuron pop_B'].to_numpy
tsB = events_B['event time pop_B'].to_numpy
frequency = pd.read_csv(trial_directory+notes+'_frequency.csv')
t = frequency['time'].to_numpy
A_N_A = frequency['activity (Hz) pop_A'].to_numpy
B_N_B = frequency['activity (Hz) pop_B'].to_numpy
stimuli = pd.read_csv(trial_directory+notes+'_stimuli.csv')
stimulus_A = stimuli['stimulus pop A']
stimulus_B = stimuli['stimulus pop B']
sum_stimulus_A = stimuli['integral stim pop A']
sum_stimulus_B = stimuli['integral stim pop B']

fig = None
ax_raster = None
ax_rate = None
fig, (ax_raster, ax_rate, ax_stimuli) = plt.subplots(3, 1, sharex=True, figsize=(5,5), dpi=300)
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
plt.xlabel("t [ms]")

fig.savefig(os.getcwd()+'/figures/'+notes+'/Figure1.eps' , bbox_inches='tight')

    

#%% FIGURE 2
#TODO 2 coint toss inuts --> mean and time integration 
#TODO 3 COIN TOSS: raster+activity+decision space
#TODO 4 n=1000 trials who's winning?+Delta S computation --> as figure3C
plt.hist(delta_s_A_winner, histtype = 'step', color = 'red', linewidth = 2)
plt.hist(delta_s_B_winner, histtype = 'step', color = 'blue', linewidth = 2)
plt.xlabel('Time integral of $s_1$(t) - $s_2$(t)')
plt.ylabel('Count #')
plt.show()
#TODO 5 coin toss with 0 variability in the inputs

#TODO 6 network performance and error trials Fig4a
#TODO 7 evaluate the network time courses at: 3.2%, 6.4%,12.8%, 25.6% Fig4b (n trials=1000 and take the mean)

#TODO 8 compare 0.0% and 51.2% --> time that it take to cross the 15Hz threshold (figure 5a-b) --> need to find a linear relationship between mean reaction time and log coherence level

#TODO 9 test stimulus duration fig6A

#TODO 10 test persistent activity --> decrease recurrent exc weights -> again 0 12.8 51.2
# da notare: no ramping, no winner takes all a 0.0, no persistent activity.

#TODO 11 testare NMDA slow reverberation --> switch off?

#TODO 12 reverse decision --> possibilità di cambiare quando avviene lo stimolo reverse (Percentage choices for A and B as function of the onset time of reversal.Fig8A Even when the signal is reversed 1 s after the stimulus onset, the decision is still re- versable by a more powerful input. Percent- age choices for A and B as function of the coherence level of the reversed signalFig8B
#coherence above 70%–80%
