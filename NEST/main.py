import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nest
import nest.raster_plot
import os
from simulate_network import *

nest.ResetKernel()
dt = 0.1
dt_rec = 10.0
nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
t0 = nest.GetKernelStatus('time')

notes = 'o400_c0_'
coherence = 0.0
order = 400
simtime = 3000.0
start_stim = 500.0
end_stim = 1000.0
current_path = os.getcwd()+'/'

results = simulate_network(coherence, order , start_stim , end_stim , simtime)     

#TODO test EXC without NMDA --> solo AMPA (no recurrent)
smA = nest.GetStatus(results["spike_monitor_A"])[0]
rmA = nest.GetStatus(results["rate_monitor_A"])[0]	
smB = nest.GetStatus(results["spike_monitor_B"])[0]
rmB = nest.GetStatus(results["rate_monitor_B"])[0]
fig = None
ax_raster = None
ax_rate = None
fig, (ax_raster, ax_rate) = plt.subplots(2, 1, sharex=True, figsize=(16,9))
plt.suptitle('Coherence ' + str(coherence*100) + '%')
evsA = smA["events"]["senders"]
tsA = smA["events"]["times"]
ax_raster.plot(tsA, evsA, ".", color='red', label ='pop A')
evsB = smB["events"]["senders"]
tsB = smB["events"]["times"]
ax_raster.plot(tsB, evsB, ".", color='blue', label ='pop B')
ax_raster.set_ylabel("neuron #")
ax_raster.set_title("Raster Plot ", fontsize=10)
ax_raster.legend()
t = np.arange(0., simtime, dt_rec)
A_N_A = np.ones((t.size, 1)) * np.nan
trmA = rmA["events"]["times"]
trmA = trmA * dt - t0
bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
A_N_A = np.histogram(trmA, bins=bins)[0] / 400 / dt_rec
ax_rate.plot(t, A_N_A * 1000, color='red', label ='pop A')
ax_rate.fill_between(t, A_N_A * 1000, color='red')
B_N_B = np.ones((t.size, 1)) * np.nan
trmB = rmB["events"]["times"]
trmB = trmB * dt - t0
bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
B_N_B = np.histogram(trmB, bins=bins)[0] / 400 / dt_rec
ax_rate.plot(t, B_N_B * 1000, color='blue', label ='pop B')
ax_rate.fill_between(t, B_N_B * 1000, color='blue')
ax_rate.vlines(start_stim, 0, 40, color='grey')
ax_rate.vlines(end_stim, 0, 40, color='grey')
ax_rate.set_ylabel("A(t) [Hz]")
ax_rate.set_title("Activity", fontsize=10)
ax_rate.legend()
plt.xlabel("t [ms]")

decisional_space = plt.figure(figsize = [10,10])
if np.mean(A_N_A*1000)>np.mean(B_N_B*1000):
    c='red'
else:
    c='blue'
plt.plot(A_N_A * 1000,B_N_B * 1000, color=c)
plt.plot([0,40],[0,40], color='grey')
plt.xlim(-0.1,40)
plt.ylim(-0.1,40)
plt.show()

#TODO save results correctly
results = pd.DataFrame(results)
results.to_csv(notes+'results.csv')

#TODO
