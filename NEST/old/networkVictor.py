"""
Implementation of a decision making model of
[1] Wang, Xiao-Jing. "Probabilistic decision making by slow reverberation in cortical circuits."
Neuron 36.5 (2002): 955-968.
Some parts of this implementation are inspired by material from
*Stanford University, BIOE 332: Large-Scale Neural Modeling, Kwabena Boahen & Tatiana Engel, 2013*,
online available.                           REVISAAAAR
"""

# Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski.
# Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
# Cambridge University Press, 2014.               REFERENCEEE BRIAN


import nest
import pylab
from random import sample
import numpy.random as rnd
#from neurodynex3.tools import plot_tools
import numpy as np
import matplotlib.pyplot as plt
from math import floor
import time
import statistics as stats
import nest.raster_plot
from nest import voltage_trace

nest.ResetKernel()
dt = 0.10
dt_rec = 1.
nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
t0 = nest.GetKernelStatus('time')

def sim_decision_making_network(N_Excit=1600, N_Inhib=400, weight_scaling_factor=1.275, t_stimulus_start=100, t_stimulus_duration=9999, coherence_level=0, stimulus_update_interval=50, mu0_mean_stimulus_Hz=80., stimulus_std_Hz=10., N_extern=800, firing_rate_extern=3.0, w_pos=1.70, f_Subpop_size=0.15, max_sim_time=10000., stop_condition_rate=None,monitored_subset_size=256):

	"""
	Args:
	-	N_Excit (int): total number of neurons in the excitatory population
        -	N_Inhib (int): nr of neurons in the inhibitory populations
        -	weight_scaling_factor: When increasing the number of neurons by 2, the weights should be scaled down by 1/2           ¿¿¿???
        -	t_stimulus_start (Quantity): time when the stimulation starts
        -	t_stimulus_duration (Quantity): duration of the stimulation
        -	coherence_level (int): coherence of the stimulus. Difference in mean between the PoissonGroups "left" stimulus and "right" stimulus
        -	stimulus_update_interval (Quantity): the mean of the stimulating PoissonGroups is re-sampled at this interval
        -	mu0_mean_stimulus_Hz (float): maximum mean firing rate of the stimulus if c=+1 or c=-1. Each neuron in the populations "Left" and "Right" receives an independent poisson input.
        -	stimulus_std_Hz (float): std deviation of the stimulating PoissonGroups.
        -	N_extern (int): nr of neurons in the stimulus independent poisson background population			¿¿¿???
        -	firing_rate_extern (int): firing rate of the stimulus independent poisson background population		¿¿¿???
        -	w_pos (float): Scaling (strengthening) of the recurrent weights within the subpopulations "Left" and "Right"
        -	f_Subpop_size (float): fraction of the neurons in the subpopulations "Left" and "Right". #left = #right = int(f_Subpop_size*N_Excit).
        -	max_sim_time (Quantity): simulated time.              							¿¿¿???
        -	stop_condition_rate (Quantity): An optional stopping criteria: If not None, the simulation stops if the firing rate of either subpopulation "Left" or "Right" is above stop_condition_rate.
        -	monitored_subset_size (int): max nr of neurons for which a state monitor is registered.			¿¿¿???
	Returns:
        -	A dictionary with the following keys (strings):
        	"rate_monitor_A", "spike_monitor_A", "voltage_monitor_A", "idx_monitored_neurons_A", 			"rate_monitor_B","spike_monitor_B", "voltage_monitor_B", "idx_monitored_neurons_B", 			"rate_monitor_Z","spike_monitor_Z","voltage_monitor_Z", "idx_monitored_neurons_Z", 			"rate_monitor_inhib","spike_monitor_inhib","voltage_monitor_inhib", "idx_monitored_neurons_inhib"
   	"""
	
	startbuild = time.time()

	print("simulating {} neurons. Start: {}".format(N_Excit + N_Inhib, time.ctime()))
	
	t_stimulus_end = t_stimulus_start + t_stimulus_duration

###############################################################################################

	### Excitatory subpopulations specification

	N_group_A = int(N_Excit * f_Subpop_size) # size of the excitatory subpopulation sensitive to stimulus A
	N_group_B = N_group_A # size of the excitatory subpopulation sensitive to stimulus B
	N_group_Z = N_Excit - N_group_A - N_group_B # (1-2f)Ne excitatory neurons do not respond to either stimulus

	Cm_excit = 500.0 # (pF) membrane capacitance of excitatory neurons (en Brian esta en nF)
	G_leak_excit = 25.0 # (nS) leak conductance
	E_leak_excit = -70.0 # (mV) reversal potential
	v_spike_thr_excit = -50.0 # (mV) spike condition
	v_reset_excit = -55.0 # (mV) reset voltage after spike
	t_abs_refract_excit = 2. # (ms) absolute refractory period

	print("Excitatory populations specified")

###############################################################################################

	### Inhibitory population specification

	Cm_inhib = 200.0 # (pF) membrane capacitance of excitatory neurons (en Brian esta en nF)
	G_leak_inhib = 20.0 # (nS) leak conductance
	E_leak_inhib = -70.0 # (mV) reversal potential
	v_spike_thr_inhib = -50.0 # (mV) spike condition
	v_reset_inhib = -55.0 # (mV) reset voltage after spike
	t_abs_refract_inhib = 1.0 # (ms) absolute refractory period

	print("Inhibitory populations specified")

###############################################################################################

	### AMPA synapses specifications

	E_AMPA = 0.0 # (mV)
	tau_AMPA_decay = 2.0 # (ms)
	tau_AMPA_rise = 0.2 # (ms)

	### GABA synapses specifications

	E_GABA = -70.0 # (mV)
	tau_GABA_decay = 5.0 # (ms)
	tau_GABA_rise = 0.25 # (ms)

	### NMDA synapses specifications

	E_NMDA = 0.0 # (mV)
	tau_NMDA_decay = 100.0 # (ms) s
	tau_NMDA_rise = 2.0 # (ms) x
	alpha_NMDA = 0.5 # (kHz)

	### projections from the external population

	g_AMPA_extern2inhib = 1.62 # (nS)
	g_AMPA_extern2excit = 2.1 # (nS)

	### projections from the inhibitory population

	g_GABA_inhib2inhib = weight_scaling_factor * 1.0 # (nS)
	g_GABA_inhib2excit = weight_scaling_factor * 1.3 # (nS)

	### projections from the excitatory population

	g_AMPA_excit2excit = weight_scaling_factor * 0.05 # (nS)
	g_AMPA_excit2inhib = weight_scaling_factor * 0.04 # (nS)
	g_NMDA_excit2excit = weight_scaling_factor * 0.165 # (nS)
	g_NMDA_excit2inhib = weight_scaling_factor * 0.13 # (nS)

###############################################################################################

	### Specify the inhibitory population

        # Define inhibitory population

	inhib_neuron_params = { "C_m": Cm_inhib,
				"g_L" : G_leak_inhib,
				"E_L" : E_leak_inhib,
				"V_th" : v_spike_thr_inhib,
				"V_reset" : v_reset_inhib,
				"t_ref" : t_abs_refract_inhib,
				"tau_rise_ex": tau_NMDA_rise,
				"tau_decay_ex": tau_NMDA_decay,
				"tau_rise_in": tau_GABA_rise,
				"tau_decay_in": tau_GABA_decay, 
				"E_ex": E_NMDA,
				"E_in": E_GABA}
	
	nest.CopyModel("iaf_cond_beta", "inhib_iaf_cond_beta", params=inhib_neuron_params)

	inhib_pop = nest.Create("inhib_iaf_cond_beta", N_Inhib)
	inhib_pop_vm = rnd.uniform (v_spike_thr_inhib -4.0, high=v_spike_thr_inhib -1.0, size=N_Inhib)
	mean_vm_in = stats.mean(inhib_pop_vm)
	print("Mean Vm of pop I: {}".format(mean_vm_in))
	nest.SetStatus(inhib_pop, "V_m", inhib_pop_vm)

	print("Inhibitory populations created")

###############################################################################################

	### Specify the excitatory population

	#Define 3 excitatory populations 

	excit_neuron_params = { "C_m": Cm_excit,
				"g_L" : G_leak_excit,
				"E_L" : E_leak_excit,
				"V_th" : v_spike_thr_excit,
				"V_reset" : v_reset_excit,
				"t_ref" : t_abs_refract_excit,
				"tau_rise_ex": tau_NMDA_rise,
				"tau_decay_ex": tau_NMDA_decay,
				"tau_rise_in": tau_GABA_rise,
				"tau_decay_in": tau_GABA_decay,
				"E_ex": E_NMDA,
				"E_in": E_GABA}

	nest.CopyModel("iaf_cond_beta", "excit_iaf_cond_beta", params=excit_neuron_params)

	excit_pop_A = nest.Create("excit_iaf_cond_beta", N_group_A)
	excit_pop_A_vm = rnd.uniform (E_leak_excit, high=E_leak_excit +5.0, size=N_group_A)
	mean_vm_A = stats.mean(excit_pop_A_vm)
	print("Mean Vm of pop A: {}".format(mean_vm_A))
	nest.SetStatus(excit_pop_A, "V_m", excit_pop_A_vm)

	excit_pop_B = nest.Create("excit_iaf_cond_beta", N_group_B)
	excit_pop_B_vm = rnd.uniform (E_leak_excit, high=E_leak_excit +5.0, size=N_group_B)
	mean_vm_B = stats.mean(excit_pop_B_vm)
	print("Mean Vm of pop B: {}".format(mean_vm_B))
	nest.SetStatus(excit_pop_B, "V_m", excit_pop_B_vm)

	excit_pop_Z = nest.Create("excit_iaf_cond_beta", N_group_Z)
	excit_pop_Z_vm = rnd.uniform (v_reset_excit, high=v_spike_thr_excit -1.0, size=N_group_Z)
	mean_vm_Z = stats.mean(excit_pop_Z_vm)
	print("Mean Vm of pop Z: {}".format(mean_vm_Z))
	nest.SetStatus(excit_pop_Z, "V_m", excit_pop_Z_vm)

	print("Excitatory populations created")

###############################################################################################

	### delay, weights and "adjusted" weights

	delay_AMPA = 0.5 # (ms)
	delay_GABA = 0.5 # (ms)
	delay_NMDA = 0.5 # (ms)

	# We use the same postsyn AMPA and NMDA conductances. Adjust the weights coming from different sources:

	w_AMPA_ext2inhib =  0.02 * (g_AMPA_extern2inhib / g_AMPA_excit2inhib)
	w_AMPA_ext2excit =  0.02 * (g_AMPA_extern2excit / g_AMPA_excit2excit)
	w0_inhib = -36.0 #(GABA)	
	w0_excit = 0.1*1.0 #(AMPA/NMDA)
	w_AMPA_pos = 0.1*w_pos
	w_AMPA_neg = 0.1*(1. - f_Subpop_size * (w_AMPA_pos - 1.) / (1. - f_Subpop_size))
	w_NMDA_pos = 0.1*w_pos
	w_NMDA_neg = 0.1*(1. - f_Subpop_size * (w_NMDA_pos - 1.) / (1. - f_Subpop_size))
	
	print("w_AMPA_ext2inhib={}, w_AMPA_ext2excit={}, w0_inhib={}, w0_excit={}, w_AMPA_pos={}, w_AMPA_neg={}, w_NMDA_pos={}, w_NMDA_neg={}".format(w_AMPA_ext2inhib, w_AMPA_ext2excit, w0_inhib, w0_excit, w_AMPA_pos, w_AMPA_neg, w_NMDA_pos, w_NMDA_neg))

###############################################################################################	
	
	### Definition of connections

	nest.CopyModel("static_synapse", "noise2pops", {"delay": delay_AMPA})
	syn_dict_inhib = {"model": "noise2pops", "weight": w_AMPA_ext2inhib}
	syn_dict_excit = {"model": "noise2pops", "weight": w_AMPA_ext2excit}

	nest.CopyModel("static_synapse", "inhibitory_GABA", {"weight" :w0_inhib, "delay": delay_GABA})

	nest.CopyModel("static_synapse", "standard_AMPA", {"weight" :w0_excit, "delay": delay_AMPA})

	nest.CopyModel("static_synapse", "standard_NMDA", {"weight" :w0_excit, "delay": delay_NMDA})

	nest.CopyModel("static_synapse", "excit_AMPA", {"weight" :w_AMPA_pos, "delay": delay_AMPA})

	nest.CopyModel("static_synapse", "excit_NMDA", {"weight" :w_NMDA_pos, "delay": delay_NMDA})

	nest.CopyModel("static_synapse", "inhib_AMPA", {"weight" :w_AMPA_neg, "delay": delay_AMPA})

	nest.CopyModel("static_synapse", "inhib_NMDA", {"weight" :w_NMDA_neg, "delay": delay_NMDA})

	print("Establishing connections")

###############################################################################################

        # projections FROM EXTERNAL POISSON GROUP (noise):

	noise = nest.Create("poisson_generator", N_extern)
	nest.SetStatus(noise, "rate", firing_rate_extern)

	nest.Connect(noise, inhib_pop, syn_spec=syn_dict_inhib)
	nest.Connect(noise, excit_pop_A, syn_spec=syn_dict_excit)
	nest.Connect(noise, excit_pop_B, syn_spec=syn_dict_excit)
	nest.Connect(noise, excit_pop_Z, syn_spec=syn_dict_excit)

###############################################################################################

        # GABA projections FROM INHIBITORY population:

	nest.Connect(inhib_pop, inhib_pop, syn_spec="inhibitory_GABA")
	nest.Connect(inhib_pop, excit_pop_A, syn_spec="inhibitory_GABA")
	nest.Connect(inhib_pop, excit_pop_B, syn_spec="inhibitory_GABA")
	nest.Connect(inhib_pop, excit_pop_Z, syn_spec="inhibitory_GABA")

###############################################################################################
	
        # AMPA projections FROM EXCITATORY A:

	nest.Connect(excit_pop_A, inhib_pop, syn_spec="standard_AMPA")
	nest.Connect(excit_pop_A, excit_pop_A, syn_spec="excit_AMPA")
	nest.Connect(excit_pop_A, excit_pop_B, syn_spec="inhib_AMPA")
	nest.Connect(excit_pop_A, excit_pop_Z, syn_spec="standard_AMPA")

###############################################################################################

	# AMPA projections FROM EXCITATORY B:

	nest.Connect(excit_pop_B, inhib_pop, syn_spec="standard_AMPA")
	nest.Connect(excit_pop_B, excit_pop_A, syn_spec="inhib_AMPA")
	nest.Connect(excit_pop_B, excit_pop_B, syn_spec="excit_AMPA")
	nest.Connect(excit_pop_B, excit_pop_Z, syn_spec="standard_AMPA")

###############################################################################################

	# AMPA projections FROM EXCITATORY Z:

	nest.Connect(excit_pop_Z, inhib_pop, syn_spec="standard_AMPA")
	nest.Connect(excit_pop_Z, excit_pop_A, syn_spec="inhib_AMPA")
	nest.Connect(excit_pop_Z, excit_pop_B, syn_spec="inhib_AMPA")
	nest.Connect(excit_pop_Z, excit_pop_Z, syn_spec="standard_AMPA")
	
###############################################################################################

	# NMDA projections FROM EXCITATORY to INHIB, A,B,Z
	
	nest.Connect(excit_pop_A, inhib_pop, syn_spec="standard_NMDA")
	nest.Connect(excit_pop_B, inhib_pop, syn_spec="standard_NMDA")
	nest.Connect(excit_pop_Z, inhib_pop, syn_spec="standard_NMDA")

	nest.Connect(excit_pop_A, excit_pop_A, syn_spec="excit_NMDA")
	nest.Connect(excit_pop_B, excit_pop_A, syn_spec="inhib_NMDA")
	nest.Connect(excit_pop_Z, excit_pop_A, syn_spec="inhib_NMDA")

	nest.Connect(excit_pop_A, excit_pop_B, syn_spec="inhib_NMDA")
	nest.Connect(excit_pop_B, excit_pop_B, syn_spec="excit_NMDA")	
	nest.Connect(excit_pop_Z, excit_pop_B, syn_spec="inhib_NMDA")

	nest.Connect(excit_pop_A, excit_pop_Z, syn_spec="standard_NMDA")
	nest.Connect(excit_pop_B, excit_pop_Z, syn_spec="standard_NMDA")
	nest.Connect(excit_pop_Z, excit_pop_Z, syn_spec="standard_NMDA")

	print("Connections established")

###############################################################################################

	# Define the stimulus: two PoissonInput with time-dependent mean.

	poissonStimulus2A = nest.Create("poisson_generator", N_group_A)
	poissonStimulus2B = nest.Create("poisson_generator", N_group_B)
	nest.CopyModel("static_synapse", "poissonStimulus", {"weight": w_AMPA_ext2excit})
	nest.Connect(poissonStimulus2A, excit_pop_A, 'one_to_one', syn_spec="poissonStimulus")
	nest.Connect(poissonStimulus2B, excit_pop_B, 'one_to_one', syn_spec="poissonStimulus")

	print("Stimulus created and connected")

	def update_poisson_stimulus(t):
		if t >= t_stimulus_start and t < t_stimulus_end:
			offset_A = mu0_mean_stimulus_Hz * (0.5 + (0.5 * coherence_level))
			offset_B = mu0_mean_stimulus_Hz * (0.5 - (0.5 * coherence_level))
	
			rate_A = np.random.normal(offset_A, stimulus_std_Hz)
			rate_A = (max(0., rate_A)) #no negative rate
			rate_B = np.random.normal(offset_B, stimulus_std_Hz)
			rate_B = (max(0., rate_B)) #no negative rate

			nest.SetStatus(poissonStimulus2A, "rate", rate_A)
			nest.SetStatus(poissonStimulus2B, "rate", rate_B)
			print("stim on. rate_A={}, rate_B={}".format(rate_A, rate_B))

		else:
			nest.SetStatus(poissonStimulus2A, "rate", 0.)
			nest.SetStatus(poissonStimulus2B, "rate", 0.)
			print("stim off.")

###############################################################################################

	def get_monitors(pop, monitored_subset_size):
		
		"""Internal helper.
		Args:
			pop: target population of which we record
			monitored_subset_size: max nr of neurons for which a state monitor is registered.
		Returns: monitors for rate, voltage, spikes and monitored neurons indexes.
		"""
		monitored_subset_size = min(monitored_subset_size, len(pop))
		print("Number of monitored neurons = {}".format(monitored_subset_size))

		#length = len(pop)
		#min_id = nest.GetStatus(pop, {"global_id"})[0]
		#max_id = nest.GetStatus(pop, {"global_id"})[length-1]
		
		idx_monitored_neurons = tuple(sample(list(pop), monitored_subset_size))

		#print(idx_monitored_neurons)

		rate_monitor = nest.Create("spike_detector")
		nest.SetStatus(rate_monitor, {'withgid': False, 'withtime': True, 'time_in_steps': True})
		nest.SetDefaults('static_synapse', {'weight': 1., 'delay': dt})
		nest.Connect(idx_monitored_neurons, rate_monitor)

		spike_monitor = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "to_file": False})
		nest.Connect(idx_monitored_neurons, spike_monitor)

		voltage_monitor = nest.Create("multimeter")
		nest.SetStatus(voltage_monitor, {"withtime": True, "record_from":["V_m"], "to_file": False})
		nest.Connect(voltage_monitor, idx_monitored_neurons)

		return rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons

###############################################################################################

	# data collection of a subset of neurons:

	rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib = get_monitors(inhib_pop, monitored_subset_size)

	rate_monitor_A, spike_monitor_A, voltage_monitor_A, idx_monitored_neurons_A = get_monitors(excit_pop_A, monitored_subset_size)

	rate_monitor_B, spike_monitor_B, voltage_monitor_B, idx_monitored_neurons_B = get_monitors(excit_pop_B, monitored_subset_size)

	rate_monitor_Z, spike_monitor_Z, voltage_monitor_Z, idx_monitored_neurons_Z = get_monitors(excit_pop_Z, monitored_subset_size)

	print("Monitors created and connected")

###############################################################################################

	endbuild = time.time()

	sim_steps = np.arange(0, max_sim_time, stimulus_update_interval)

	print(sim_steps)

	for i, step in enumerate(sim_steps):
		print("Step number {} of {}".format(i+1, len(sim_steps)))
		update_poisson_stimulus(step)
		nest.Simulate(stimulus_update_interval)

	"""if stop_condition_rate is None:
		nest.Simulate(max_sim_time)
	else:
		sim_sum = 0
		sim_batch = 100.
		samples_in_batch = int(floor(sim_batch / dt))
		avg_rate_in_batch = 0
		while (sim_sum < max_sim_time) and (avg_rate_in_batch < stop_condition_rate):
			nest.Simulate(sim_batch)
			### en NEST se accede distinto a la rate del monitor
			avg_A = np.mean(rate_monitor_A.rate[-samples_in_batch:])
			avg_B = np.mean(rate_monitor_B.rate[-samples_in_batch:])
			avg_rate_in_batch = max(avg_A, avg_B)
			sim_sum += sim_batch
	"""

	endsimulate = time.time()

	print("sim end. {}".format(time.ctime()))
	ret_vals = dict()

	ret_vals["rate_monitor_A"] = rate_monitor_A
	ret_vals["spike_monitor_A"] = spike_monitor_A
	ret_vals["voltage_monitor_A"] = voltage_monitor_A
	ret_vals["idx_monitored_neurons_A"] = idx_monitored_neurons_A

	ret_vals["rate_monitor_B"] = rate_monitor_B
	ret_vals["spike_monitor_B"] = spike_monitor_B
	ret_vals["voltage_monitor_B"] = voltage_monitor_B
	ret_vals["idx_monitored_neurons_B"] = idx_monitored_neurons_B

	ret_vals["rate_monitor_Z"] = rate_monitor_Z
	ret_vals["spike_monitor_Z"] = spike_monitor_Z
	ret_vals["voltage_monitor_Z"] = voltage_monitor_Z
	ret_vals["idx_monitored_neurons_Z"] = idx_monitored_neurons_Z

	ret_vals["rate_monitor_inhib"] = rate_monitor_inhib
	ret_vals["spike_monitor_inhib"] = spike_monitor_inhib
	ret_vals["voltage_monitor_inhib"] = voltage_monitor_inhib
	ret_vals["idx_monitored_neurons_inhib"] = idx_monitored_neurons_inhib

	ret_vals["weights"] = (w_AMPA_ext2inhib, w_AMPA_ext2excit, w0_inhib, w0_excit, w_AMPA_pos, w_AMPA_neg, w_NMDA_pos, w_NMDA_neg)
	ret_vals["delays"] = (delay_AMPA, delay_GABA, delay_NMDA)

	build_time = endbuild - startbuild
	sim_time = endsimulate - endbuild
	
	print("Number of neurons : {0}".format(N_Excit+N_Inhib))
	print("Building time     : %.2f s" % build_time)
	print("Simulation time   : %.2f s" % sim_time)

	return ret_vals

###############################################################################################

# def run_multiple_simulations

###############################################################################################

def print_version():
	print("Version: 21 March 2021")	

###############################################################################################

def getting_started():

	"""
	A simple example to get started.
	Returns:
	"""
	stim_start = 150.
	stim_duration = 350. 
	max_sim_time = 800.
	coherence_level = 0.70
	firing_rate_extern = 3.0
	N_extern = 800
	mu0_mean_stimulus_Hz = 80.
	w_pos = 1.7

	print("stimulus start {}, stimulus end: {}".format(stim_start, stim_start+stim_duration))
	
	results = sim_decision_making_network(N_Excit=1600, N_Inhib=400, weight_scaling_factor=1.275, t_stimulus_start=stim_start, t_stimulus_duration=stim_duration, coherence_level=coherence_level, stimulus_update_interval=50, mu0_mean_stimulus_Hz=mu0_mean_stimulus_Hz, stimulus_std_Hz=10.,N_extern=N_extern, firing_rate_extern=firing_rate_extern, w_pos=w_pos, f_Subpop_size=0.15, max_sim_time=max_sim_time, monitored_subset_size=256)
	
	"""###plotting with module packages
	nest.raster_plot.from_device(results["spike_monitor_A"], hist=True)
	plt.title('Population A dynamics')

	plt.figure(2)
	voltage_trace.from_device(results["voltage_monitor_A"])
	plt.title('Voltage trace A')

	nest.raster_plot.from_device(results["spike_monitor_B"], hist=True)
	plt.title('Population B dynamics')

	pylab.figure(4)
	voltage_trace.from_device(results["voltage_monitor_B"])
	plt.title('Voltage trace B')

	nest.raster_plot.from_device(results["spike_monitor_Z"], hist=True)
	plt.title('Population Z dynamics')

	pylab.figure(6)
	voltage_trace.from_device(results["voltage_monitor_Z"])
	plt.title('Voltage trace Z')

	nest.raster_plot.from_device(results["spike_monitor_inhib"], hist=True)
	plt.title('Population inhib dynamics')

	pylab.figure(8)
	voltage_trace.from_device(results["voltage_monitor_inhib"])
	plt.title('Voltage trace inhib')

	plt.show()"""

	events_A = nest.GetStatus(results["spike_monitor_A"], "n_events")[0]
	events_B = nest.GetStatus(results["spike_monitor_B"], "n_events")[0]
	events_I = nest.GetStatus(results["spike_monitor_inhib"], "n_events")[0]
	rate_A = events_A / max_sim_time * 1000.0 / len(results["idx_monitored_neurons_A"])
	rate_B = events_B / max_sim_time * 1000.0 / len(results["idx_monitored_neurons_B"])
	rate_I = events_I / max_sim_time * 1000.0 / len(results["idx_monitored_neurons_inhib"])

	print("Population A rate   : %.2f Hz" % rate_A)
	print("Population B rate   : %.2f Hz" % rate_B)
	print("Population Inhib rate   : %.2f Hz" % rate_I)

	"""print("w_AMPA_ext2inhib: {}".format(results["weights"][0]))
	print("w_AMPA_ext2excit: {}".format(results["weights"][1]))
	print("w0_inhib: {}".format(results["weights"][2]))
	print("w0_excit: {}".format(results["weights"][3]))
	print("w_AMPA_pos: {}".format(results["weights"][4]))
	print("w_AMPA_neg: {}".format(results["weights"][5]))
	print("w_NMDA_pos: {}".format(results["weights"][6]))
	print("w_NMDA_neg: {}".format(results["weights"][7]))

	print("delay_AMPA: {}".format(results["delays"][0]))
	print("delay_GABA: {}".format(results["delays"][1]))
	print("delay_NMDA: {}".format(results["delays"][2]))"""

	print(results["weights"][0])
	print(results["weights"][1])
	print(results["weights"][2])
	print(results["weights"][3])
	print(results["weights"][4])
	print(results["weights"][5])
	print(results["weights"][6])
	print(results["weights"][7])

	print(results["delays"][0])
	print(results["delays"][1])
	print(results["delays"][2])

	print(coherence_level)
	print(firing_rate_extern)
	print(N_extern)
	print(mu0_mean_stimulus_Hz)
	print(w_pos)
	print(rate_A)
	print(rate_B)
	print(rate_I)
	### plotting without module packages

	vmA = nest.GetStatus(results["voltage_monitor_A"])[0]
	smA = nest.GetStatus(results["spike_monitor_A"])[0]
	rmA = nest.GetStatus(results["rate_monitor_A"])[0]	

	vmB = nest.GetStatus(results["voltage_monitor_B"])[0]
	smB = nest.GetStatus(results["spike_monitor_B"])[0]
	rmB = nest.GetStatus(results["rate_monitor_B"])[0]

	#A

	fig = None
	ax_raster = None
	ax_rate = None
	ax_voltage = None
	fig, (ax_raster, ax_rate, ax_voltage) = plt.subplots(3, 1, sharex=True, figsize=(10,5))
	
	plt.suptitle("Left (population A)")

	evsA = smA["events"]["senders"]
	tsA = smA["events"]["times"]
	ax_raster.plot(tsA, evsA, ".")
	ax_raster.set_ylabel("neuron #")
	ax_raster.set_title("Raster Plot", fontsize=10)
	
	t = np.arange(0., max_sim_time, dt_rec)
	A_N_A = np.ones((t.size, 1)) * np.nan
	trmA = rmA["events"]["times"]
	trmA = trmA * dt - t0
	bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
	A_N_A = np.histogram(trmA, bins=bins)[0] / len(results["idx_monitored_neurons_A"]) / dt_rec
	ax_rate.plot(t, A_N_A * 1000)
	ax_rate.set_ylabel("A(t) [Hz]")
	ax_rate.set_title("Activity", fontsize=10)

	for i in results["idx_monitored_neurons_A"]:
		per=len(results["idx_monitored_neurons_A"])
		#print("Neuron nº: {}".format(i))
		VmsA = vmA["events"]["V_m"][i::per]
		tsA = vmA["events"]["times"][i::per]
		ax_voltage.plot(tsA, VmsA)
	ax_voltage.set_ylabel("V(t) [mV]")
	ax_voltage.set_title("Voltage traces", fontsize=10)
	ax_voltage.set_ylim([-72.0, -48.0])
	plt.xlabel("t [ms]")

	#B

	fig = None
	ax_raster = None
	ax_rate = None
	ax_voltage = None
	fig, (ax_raster, ax_rate, ax_voltage) = plt.subplots(3, 1, sharex=True, figsize=(10,5))
	
	plt.suptitle("Right (population B)")

	evsB = smB["events"]["senders"]
	tsB = smB["events"]["times"]
	ax_raster.plot(tsB, evsB, ".")
	ax_raster.set_ylabel("neuron #")
	ax_raster.set_title("Raster Plot", fontsize=10)
	
	t = np.arange(0., max_sim_time, dt_rec)
	A_N_B = np.ones((t.size, 1)) * np.nan
	trmB = rmB["events"]["times"]
	trmB = trmB * dt - t0
	bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
	A_N_B = np.histogram(trmB, bins=bins)[0] / len(results["idx_monitored_neurons_B"]) / dt_rec
	ax_rate.plot(t, A_N_B * 1000)
	ax_rate.set_ylabel("A(t) [Hz]")
	ax_rate.set_title("Activity", fontsize=10)

	for i in results["idx_monitored_neurons_B"]:
		per=len(results["idx_monitored_neurons_B"])
		#print("Neuron nº: {}".format(i))
		VmsB = vmB["events"]["V_m"][i::per]
		tsB = vmB["events"]["times"][i::per]
		ax_voltage.plot(tsB, VmsB)
	ax_voltage.set_ylabel("V(t) [mV]")
	ax_voltage.set_title("Voltage traces", fontsize=10)
	ax_voltage.set_ylim([-72.0, -48.0])
	plt.xlabel("t [ms]")

	"""#Z

	vmZ = nest.GetStatus(results["voltage_monitor_Z"])[0]
	smZ = nest.GetStatus(results["spike_monitor_Z"])[0]
	rmZ = nest.GetStatus(results["rate_monitor_Z"])[0]

	fig = None
	ax_raster = None
	ax_rate = None
	ax_voltage = None
	fig, (ax_raster, ax_rate, ax_voltage) = plt.subplots(3, 1, sharex=True, figsize=(10,5))
	
	plt.suptitle("Non selective (population Z)")

	evsZ = smZ["events"]["senders"]
	tsZ = smZ["events"]["times"]
	ax_raster.plot(tsZ, evsZ, ".")
	ax_raster.set_ylabel("neuron #")
	ax_raster.set_title("Raster Plot", fontsize=10)

	t = np.arange(0., max_sim_time, dt_rec)
	A_N_Z = np.ones((t.size, 1)) * np.nan
	trmZ = rmZ["events"]["times"]
	trmZ = trmZ * dt - t0
	bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
	A_N_Z = np.histogram(trmZ, bins=bins)[0] / len(results["idx_monitored_neurons_Z"]) / dt_rec
	ax_rate.plot(t, A_N_Z * 1000)
	ax_rate.set_ylabel("A(t) [Hz]")
	ax_rate.set_title("Activity", fontsize=10)

	for i in results["idx_monitored_neurons_Z"]:
		per=len(results["idx_monitored_neurons_Z"])
		#print("Neuron nº: {}".format(i))
		VmsZ = vmZ["events"]["V_m"][i::per]
		tsZ = vmZ["events"]["times"][i::per]
		ax_voltage.plot(tsZ, VmsZ)
	ax_voltage.set_ylabel("V(t) [mV]")
	ax_voltage.set_title("Voltage traces", fontsize=10)
	plt.xlabel("t [ms]")"""

	#Inhib

	vmI = nest.GetStatus(results["voltage_monitor_inhib"])[0]
	smI = nest.GetStatus(results["spike_monitor_inhib"])[0]
	rmI = nest.GetStatus(results["rate_monitor_inhib"])[0]

	fig = None
	ax_raster = None
	ax_rate = None
	ax_voltage = None
	fig, (ax_raster, ax_rate, ax_voltage) = plt.subplots(3, 1, sharex=True, figsize=(10,5))
	
	plt.suptitle("Inhibitory")

	evsI = smI["events"]["senders"]
	tsI = smI["events"]["times"]
	ax_raster.plot(tsI, evsI, ".")
	ax_raster.set_ylabel("neuron #")
	ax_raster.set_title("Raster Plot", fontsize=10)

	t = np.arange(0., max_sim_time, dt_rec)
	A_N_I = np.ones((t.size, 1)) * np.nan
	trmI = rmI["events"]["times"]
	trmI = trmI * dt - t0
	bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
	A_N_I = np.histogram(trmI, bins=bins)[0] / len(results["idx_monitored_neurons_inhib"]) / dt_rec
	ax_rate.plot(t, A_N_I * 1000)
	ax_rate.set_ylabel("A(t) [Hz]")
	ax_rate.set_title("Activity", fontsize=10)

	for i in results["idx_monitored_neurons_inhib"]:
		per=len(results["idx_monitored_neurons_inhib"])
		#print("Neuron nº: {}".format(i))
		VmsI = vmI["events"]["V_m"][i::per]
		tsI = vmI["events"]["times"][i::per]
		ax_voltage.plot(tsI, VmsI)
	ax_voltage.set_ylabel("V(t) [mV]")
	ax_voltage.set_title("Voltage traces", fontsize=10)
	plt.xlabel("t [ms]")

	plt.show()

if __name__ == "__main__":
	getting_started()


