import numpy as np
from random import sample
import numpy.random as rnd
from math import floor
import time
import matplotlib.pyplot as plt
import statistics as stats
import os
import shutil
import sys 
import nest
import nest.raster_plot
import pandas as pd

nest.ResetKernel()
dt = 0.1
dt_rec = 10.0
nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
t0 = nest.GetKernelStatus('time')

#'''
#'''**********************************************************************************
def LambertWm1(x):
    return nest.ll_api.sli_func('LambertWm1', float(x))

def ComputePSPNorm(tau_mem, C_mem, tau_syn):
    a = (tau_mem / tau_syn)
    b = (1.0 / tau_syn -1.0 / tau_mem)
    t_max = 1.0 / b * (-LambertWm1(-np.exp(-1.0/a)/a) - 1.0 / a)
    return (np.exp(1.0) / (tau_syn * (C_mem * b) * 
            ((np.exp( -t_max / tau_mem) - np.exp(-t_max / tau_syn)) / b - 
            t_max * np.exp(-t_max / tau_syn)))) 

def simulate_network_revstim(t_rev=1000., stim_rev=-0.8, n_run=1,coherence = 51.2, order = 400, start_stim = 500.0, end_stim = 1000.0, simtime = 3000.0, stimulus_update_interval = 25, fn_fixed_par = "fixed_parameters.csv", fn_tuned_par = "tuned_parameters.csv", rec_pop=1.):
    
    current_path = os.getcwd()+'/'
    fixed_pars = pd.read_csv(current_path+fn_fixed_par)
    tuned_par = pd.read_csv(current_path+fn_tuned_par)
    startbuild = time.time()
    nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
    NB = 2 * order  # number of excitatory neurons in pop B
    NA = 2 * order  # number of excitatory neurons in pop A

    NI = 1 * order  # number of inhibitory neurons
    N_neurons = NA + NB + NI   # number of neurons in total
    N_rec = order * (2+2+1)  # record from all neurons

    tau_syn = [fixed_pars['tau_syn_noise'][0],fixed_pars['tau_syn_AMPA'][0], fixed_pars['tau_syn_NMDA'][0], fixed_pars['tau_syn_GABA'][0]]  # [ms]

    exc_neuron_params = {
        "E_L": fixed_pars['V_membrane'][0],
        "V_th": fixed_pars['V_threshold'][0],
        "V_reset": fixed_pars['V_reset'][0],
        "C_m": fixed_pars['C_m_ex'][0],
        "tau_m": fixed_pars['tau_m_ex'][0],
        "t_ref": fixed_pars['t_ref_ex'][0], 
        "tau_syn": tau_syn
    }
    inh_neuron_params = {
        "E_L": fixed_pars['V_membrane'][0],
        "V_th": fixed_pars['V_threshold'][0],
        "V_reset": fixed_pars['V_reset'][0],
        "C_m": fixed_pars['C_m_in'][0],
        "tau_m": fixed_pars['tau_m_in'][0],
        "t_ref": fixed_pars['t_ref_in'][0], 
        "tau_syn": tau_syn
    }

    nest.CopyModel("iaf_psc_exp_multisynapse", "excitatory_pop", params=exc_neuron_params)
    #nest.SetDefaults("iaf_psc_exp_multisynapse", exc_neuron_params)
    #SE INVERTO ORDINE DI CREAZIONE RENDO LA POP DEFINITA PER PRIMA PIÙ SENSIBILE WTF?!?!?!? --> risolto con CopyModel
    pop_A = nest.Create("excitatory_pop", NA)
    pop_B = nest.Create("excitatory_pop", NB)

    nest.CopyModel("iaf_psc_exp_multisynapse", "inhibitory_pop", params=inh_neuron_params)
    #nest.SetDefaults("iaf_psc_exp_multisynapse", inh_neuron_params)
    pop_inh = nest.Create("inhibitory_pop", NI)

    #'''
    #'''**********************************************************************************

    J = tuned_par['J'][0]  # mV -> this means that it takes 200 simultaneous events to drive the spiking activity 

    J_unit_noise = ComputePSPNorm(fixed_pars['tau_m_ex'][0], fixed_pars['C_m_ex'][0], fixed_pars['tau_syn_noise'][0])
    J_norm_noise = J / J_unit_noise 

    J_unit_AMPA = ComputePSPNorm(fixed_pars['tau_m_ex'][0], fixed_pars['C_m_ex'][0], fixed_pars['tau_syn_AMPA'][0])
    J_norm_AMPA = J / J_unit_AMPA 

    J_norm_NMDA = 0.05  # the weight for the NMDA is set at 0.05, cannot compute J_unit_NMDA since tau_syn_NMDA is greater then tau_m_ex

    J_unit_GABA = ComputePSPNorm(fixed_pars['tau_m_in'][0], fixed_pars['C_m_in'][0], fixed_pars['tau_syn_GABA'][0])
    J_norm_GABA = J / J_unit_GABA
    #'''
    #'''**********************************************************************************

    #Recording devices
    spike_monitor_A = nest.Create("spike_detector", params={"label": "Excitatory population A","withgid": True, "withtime": True, "to_file": False})
    spike_monitor_B = nest.Create("spike_detector", params={"label": "Excitatory population B","withgid": True, "withtime": True, "to_file": False})
    spike_monitor_in = nest.Create("spike_detector", params={"label": "Inhibitory population ","withgid": True, "withtime": True, "to_file": False})
    nest.Connect(pop_A, spike_monitor_A)
    nest.Connect(pop_B, spike_monitor_B)
    nest.Connect(pop_inh, spike_monitor_in)

    rate_monitor_A = nest.Create("spike_detector")
    rate_monitor_B = nest.Create("spike_detector")
    rate_monitor_in = nest.Create("spike_detector")
    nest.SetStatus(rate_monitor_A, [{"label": "Excitatory population A", 'withgid': False, 'withtime': True, 'time_in_steps': True}])
    nest.SetStatus(rate_monitor_B, [{"label": "Excitatory population B",'withgid': False, 'withtime': True, 'time_in_steps': True}])
    nest.SetStatus(rate_monitor_in, [{"label": "Inhibitory population ",'withgid': False, 'withtime': True, 'time_in_steps': True}])
    nest.SetDefaults('static_synapse', {'weight': 1., 'delay': dt})
    nest.Connect(pop_A, rate_monitor_A)
    nest.Connect(pop_B, rate_monitor_B)
    nest.Connect(pop_inh, rate_monitor_in)

    #'''
    #'''**********************************************************************************

    # Input noise
    nu_th_noise_ex = (np.abs(fixed_pars['V_threshold'][0]) * fixed_pars['C_m_ex'][0]) / (J_norm_noise * np.exp(1) * fixed_pars['tau_m_ex'][0] * fixed_pars['tau_syn_noise'][0])
    nu_ex = tuned_par['eta_ex'][0] * nu_th_noise_ex
    p_rate_ex = 1000.0 * nu_ex

    nu_th_noise_in = (np.abs(fixed_pars['V_threshold'][0]) * fixed_pars['C_m_in'][0]) / (J_norm_noise * np.exp(1) * fixed_pars['tau_m_in'][0] * fixed_pars['tau_syn_noise'][0])
    nu_in = tuned_par['eta_in'][0] * nu_th_noise_in
    p_rate_in = 1000.0 * nu_in

    #nest.SetDefaults("poisson_generator", {"rate": p_rate_ex})    #poisson generator for the noise in input to popA and popB
    PG_noise_to_B = nest.Create("poisson_generator")
    PG_noise_to_A = nest.Create("poisson_generator")

    #nest.SetDefaults("poisson_generator", {"rate": p_rate_in})   #poisson generator for the noise in input to popinh
    PG_noise_to_inh = nest.Create("poisson_generator")

    nest.CopyModel("static_synapse", "noise_syn",
                {"weight": J_norm_noise, "delay": fixed_pars['delay_noise'][0]})
    noise_syn = {"model": "noise_syn",
                    "receptor_type": 1}

    
    nest.Connect(PG_noise_to_A, pop_A, syn_spec=noise_syn)
    nest.Connect(PG_noise_to_B, pop_B, syn_spec=noise_syn)
    nest.Connect(PG_noise_to_inh, pop_inh, syn_spec=noise_syn)

    #'''
    #'''**********************************************************************************
    
    # Input stimulus
    PG_input_AMPA_B = nest.Create("poisson_generator")
    PG_input_AMPA_A = nest.Create("poisson_generator")

    nest.CopyModel("static_synapse", "excitatory_AMPA_input",
                {"weight": J_norm_AMPA, "delay": fixed_pars['delay_AMPA'][0]})
    AMPA_input_syn = {"model": "excitatory_AMPA_input",
                    "receptor_type": 2} 
  
    nest.Connect(PG_input_AMPA_A, pop_A, syn_spec=AMPA_input_syn)
    nest.Connect(PG_input_AMPA_B, pop_B, syn_spec=AMPA_input_syn)

    # Define the stimulus: two PoissonInput with time-dependent mean.
    mean_p_rate_stimulus=  p_rate_ex / tuned_par['ratio_stim_rate'][0]   #rate for the input Poisson generator to popA (scaled with respect to the noise)
    std_p_rate_stimulus = mean_p_rate_stimulus / tuned_par['std_ratio'][0]

    def update_poisson_stimulus(t):

        rate_noise_B = np.random.normal(p_rate_ex, p_rate_ex/tuned_par['std_noise'][0])
        rate_noise_A = np.random.normal(p_rate_ex, p_rate_ex/tuned_par['std_noise'][0])
        rate_noise_inh = np.random.normal(p_rate_in, p_rate_in/tuned_par['std_noise'][0])
        nest.SetStatus(PG_noise_to_A, "rate", rate_noise_A)
        nest.SetStatus(PG_noise_to_B, "rate", rate_noise_B)
        nest.SetStatus(PG_noise_to_inh, "rate", rate_noise_inh)

        if t >= start_stim  and t < t_rev:
            offset_A = mean_p_rate_stimulus * (0.5 - (0.5 * coherence))
            offset_B = mean_p_rate_stimulus * (0.5 + (0.5 * coherence))

            rate_B = np.random.normal(offset_B, std_p_rate_stimulus)
            rate_B = (max(0., rate_B)) #no negative rate
            rate_A = np.random.normal(offset_A, std_p_rate_stimulus)
            rate_A = (max(0., rate_A)) #no negative rate
            
            #ZERO VARIABILITY (TODO 5)
            #rate_A = mean_p_rate_stimulus   
            #rate_B = mean_p_rate_stimulus
            
            nest.SetStatus(PG_input_AMPA_A, "rate", rate_A)
            nest.SetStatus(PG_input_AMPA_B, "rate", rate_B)
            print("stim on. rate_A={}, rate_B={}".format(rate_A, rate_B))
            print('trial number {}'.format(n_run))

        elif t >= t_rev  and t < end_stim:
            offset_A = mean_p_rate_stimulus * (0.5 - (0.5 * stim_rev))
            offset_B = mean_p_rate_stimulus * (0.5 + (0.5 * stim_rev))

            rate_B = np.random.normal(offset_B, std_p_rate_stimulus)
            rate_B = (max(0., rate_B)) #no negative rate
            rate_A = np.random.normal(offset_A, std_p_rate_stimulus)
            rate_A = (max(0., rate_A)) #no negative rate
            
            #ZERO VARIABILITY (TODO 5)
            #rate_A = mean_p_rate_stimulus   
            #rate_B = mean_p_rate_stimulus
            
            nest.SetStatus(PG_input_AMPA_A, "rate", rate_A)
            nest.SetStatus(PG_input_AMPA_B, "rate", rate_B)
            print("stim on. rate_A={}, rate_B={}".format(rate_A, rate_B))
            print('trial number {}'.format(n_run))
            
        else:
            nest.SetStatus(PG_input_AMPA_A, "rate", 0.)
            nest.SetStatus(PG_input_AMPA_B, "rate", 0.)

            rate_A = 0.0
            rate_B = 0.0
            print("stim off.")
            print('trial number {}'.format(n_run))
        
        return rate_A, rate_B, rate_noise_A, rate_noise_B

    #'''
    #'''**********************************************************************************

    def get_monitors(pop, monitored_subset_size):
        
        """Internal helper.
        Args:
            pop: target population of which we record
            monitored_subset_size: max nr of neurons for which a state monitor is registered.
        Returns: monitors for rate, voltage, spikes and monitored neurons indexes.
        """
        monitored_subset_size = min(monitored_subset_size, len(pop))
        
        idx_monitored_neurons = tuple(sample(list(pop), monitored_subset_size))

        rate_monitor = nest.Create("spike_detector")
        nest.SetStatus(rate_monitor, {'withgid': False, 'withtime': True, 'time_in_steps': True})
        nest.SetDefaults('static_synapse', {'weight': 1., 'delay': dt})
        nest.Connect(idx_monitored_neurons, rate_monitor)

        spike_monitor = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "to_file": False})
        nest.Connect(idx_monitored_neurons, spike_monitor)

        return rate_monitor, spike_monitor, idx_monitored_neurons

    # data collection of a subset of neurons:

    rate_monitor_A, spike_monitor_A,  idx_monitored_neurons_A = get_monitors(pop_A, int(rec_pop*len(pop_A)))
    rate_monitor_B, spike_monitor_B,  idx_monitored_neurons_B = get_monitors(pop_B, int(rec_pop*len(pop_B)))
    rate_monitor_inh, spike_monitor_inh,  idx_monitored_neurons_inh = get_monitors(pop_inh, int(rec_pop*len(pop_inh)))

    #'''
    #'''**********************************************************************************

    # Populations

    nest.CopyModel("static_synapse", "excitatory_AMPA_AB_BA",
                {"weight": J_norm_AMPA*tuned_par['w_minus'][0], "delay": fixed_pars['delay_AMPA'][0]})
    AMPA_AB_BA_syn = {"model": "excitatory_AMPA_AB_BA",
                    "receptor_type": 2}
    nest.CopyModel("static_synapse", "excitatory_NMDA_AB_BA",
                {"weight": J_norm_NMDA*tuned_par['w_minus'][0], "delay": fixed_pars['delay_NMDA'][0]})
    NMDA_AB_BA_syn = {"model": "excitatory_NMDA_AB_BA",
                    "receptor_type": 3}               

    nest.CopyModel("static_synapse", "excitatory_AMPA_AI_BI",
                {"weight": J_norm_AMPA*tuned_par['w_plus'][0], "delay": fixed_pars['delay_AMPA'][0]})
    AMPA_AI_BI_syn = {"model": "excitatory_AMPA_AI_BI",
                    "receptor_type": 2}                  
    nest.CopyModel("static_synapse", "excitatory_NMDA_AI_BI",
                {"weight": J_norm_NMDA*tuned_par['w_plus'][0], "delay": fixed_pars['delay_NMDA'][0]})
    NMDA_AI_BI_syn = {"model": "excitatory_NMDA_AI_BI",
                    "receptor_type": 3}  

    nest.CopyModel("static_synapse", "inhibitory_IA_IB",
                {"weight": -J_norm_GABA*tuned_par['w_plus'][0], "delay": fixed_pars['delay_GABA'][0]})
    GABA_IA_IB_syn = {"model": "inhibitory_IA_IB",
                    "receptor_type": 4} 

    nest.CopyModel("static_synapse", "excitatory_AMPA_recurrent",
                {"weight": J_norm_AMPA, "delay": fixed_pars['delay_AMPA'][0]})
    AMPA_recurrent_syn = {"model": "excitatory_AMPA_recurrent",
                    "receptor_type": 2}    
    nest.CopyModel("static_synapse", "excitatory_NMDA_recurrent",
                {"weight": J_norm_NMDA*tuned_par['w_plus_NMDA'][0], "delay": fixed_pars['delay_NMDA'][0]})
    NMDA_recurrent_syn = {"model": "excitatory_NMDA_recurrent",
                    "receptor_type": 3} 
    nest.CopyModel("static_synapse", "inhibitory_recurrent",
                {"weight": -J_norm_GABA, "delay": fixed_pars['delay_GABA'][0]})
    GABA_recurrent_syn = {"model": "inhibitory_recurrent",
                    "receptor_type": 4} 

    #Connecting populations
    conn_params_ex_AB_BA = {'rule': 'pairwise_bernoulli', 'p':fixed_pars['epsilon_ex_AB_BA'][0]}
    conn_params_ex_reccurent = {'rule': 'pairwise_bernoulli', 'p':fixed_pars['epsilon_ex_reccurent'][0]}
    conn_params_ex_AI_BI = {'rule': 'pairwise_bernoulli', 'p':fixed_pars['epsilon_ex_AI_BI'][0]}
    conn_params_in_IA_IB = {'rule': 'pairwise_bernoulli', 'p':fixed_pars['epsilon_in_IA_IB'][0]}
    conn_params_in_recurrent = {'rule': 'pairwise_bernoulli', 'p':fixed_pars['epsilon_in_recurrent'][0]}

    # pop A
    # Recurrent
    nest.Connect(pop_A, pop_A, conn_params_ex_reccurent, AMPA_recurrent_syn)
    nest.Connect(pop_A, pop_A, conn_params_ex_reccurent, NMDA_recurrent_syn)
    # To pop B
    nest.Connect(pop_A, pop_B, conn_params_ex_AB_BA, AMPA_AB_BA_syn)
    nest.Connect(pop_A, pop_B, conn_params_ex_AB_BA, NMDA_AB_BA_syn)
    # To pop inh.
    nest.Connect(pop_A, pop_inh, conn_params_ex_AI_BI, AMPA_AI_BI_syn)
    nest.Connect(pop_A, pop_inh, conn_params_ex_AI_BI, NMDA_AI_BI_syn)

    # pop B
    # Recurrent
    nest.Connect(pop_B, pop_B, conn_params_ex_reccurent, AMPA_recurrent_syn)
    nest.Connect(pop_B, pop_B, conn_params_ex_reccurent, NMDA_recurrent_syn)
    # To pop B
    nest.Connect(pop_B, pop_A, conn_params_ex_AB_BA, AMPA_AB_BA_syn)
    nest.Connect(pop_B, pop_A, conn_params_ex_AB_BA, NMDA_AB_BA_syn)
    # To pop inh.
    nest.Connect(pop_B, pop_inh, conn_params_ex_AI_BI, AMPA_AI_BI_syn)
    nest.Connect(pop_B, pop_inh, conn_params_ex_AI_BI, NMDA_AI_BI_syn)

    # pop inhib
    # Recurrent
    nest.Connect(pop_inh, pop_inh, conn_params_in_recurrent, GABA_recurrent_syn)
    # To pop A
    nest.Connect(pop_inh, pop_A, conn_params_in_IA_IB, GABA_IA_IB_syn)
    # To pop B
    nest.Connect(pop_inh, pop_B, conn_params_in_IA_IB, GABA_IA_IB_syn)

    #'''
    #'''**********************************************************************************

    endbuild = time.time()
    
    sim_steps = np.arange(0, simtime, stimulus_update_interval)
    print(sim_steps)

    stimulus_A = np.zeros((int(simtime)))
    stimulus_B = np.zeros((int(simtime)))

    noise_A = np.zeros((int(simtime)))
    noise_B = np.zeros((int(simtime)))


    for i, step in enumerate(sim_steps):
        print("Step number {} of {}".format(i+1, len(sim_steps)))
        rate_A, rate_B, rate_noise_A, rate_noise_B = update_poisson_stimulus(step)

        stimulus_A[int(step):int(step+stimulus_update_interval)] = rate_A
        stimulus_B[int(step):int(step+stimulus_update_interval)] = rate_B

        noise_A[int(step):int(step+stimulus_update_interval)] = rate_noise_A
        noise_B[int(step):int(step+stimulus_update_interval)] = rate_noise_B

        nest.Simulate(stimulus_update_interval)

    endsimulate = time.time()

    ret_vals = dict()

    ret_vals["rate_monitor_A"] = rate_monitor_A
    ret_vals["spike_monitor_A"] = spike_monitor_A
    ret_vals["idx_monitored_neurons_A"] = idx_monitored_neurons_A

    ret_vals["rate_monitor_B"] = rate_monitor_B
    ret_vals["spike_monitor_B"] = spike_monitor_B
    ret_vals["idx_monitored_neurons_B"] = idx_monitored_neurons_B

    ret_vals["rate_monitor_inh"] = rate_monitor_inh
    ret_vals["spike_monitor_inh"] = spike_monitor_inh
    ret_vals["idx_monitored_neurons_inh"] = idx_monitored_neurons_inh
    
    build_time = endbuild - startbuild
    sim_time = endsimulate - endbuild

    print("Building time     : %.2f s" % build_time)
    print("Simulation time   : %.2f s" % sim_time)    

    return ret_vals, stimulus_A, stimulus_B, noise_A, noise_B

def main():

    simtime = 3000.0
    start_stim = 500.0
    end_stim = 1000.0

    results, stimulus_A, stimulus_B, noise_A, noise_B = simulate_network(simtime = simtime, start_stim = start_stim, end_stim = end_stim)     

    return results, stimulus_A, stimulus_B, noise_A, noise_B

if __name__ == "__main__":
	main()