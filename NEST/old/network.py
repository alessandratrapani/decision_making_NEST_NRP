import numpy
import time
import matplotlib.pyplot as plt
import os
import shutil
import sys 
import nest
import nest.raster_plot
import pandas as pd

nest.ResetKernel()
startbuild = time.time()
#'''
#'''**********************************************************************************
def LambertWm1(x):
    return nest.ll_api.sli_func('LambertWm1', float(x))

def ComputePSPNorm(tau_mem, C_mem, tau_syn):
    a = (tau_mem / tau_syn)
    b = (1.0 / tau_syn -1.0 / tau_mem)
    t_max = 1.0 / b * (-LambertWm1(-numpy.exp(-1.0/a)/a) - 1.0 / a)
    return (numpy.exp(1.0) / (tau_syn * (C_mem * b) * 
            ((numpy.exp( -t_max / tau_mem) - numpy.exp(-t_max / tau_syn)) / b - 
            t_max * numpy.exp(-t_max / tau_syn))))
#'''
#'''**********************************************************************************
parameters = pd.read_csv('parameters.csv')
dt = 0.1   # the resolution in ms
simtime = 3000.0  # Simulation time in ms
coherence = 0.128

notes = 'diffcompstim_c128'
#'''
#'''**********************************************************************************

nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})

#'''
#'''**********************************************************************************
order = 200
NA = 2 * order  # number of excitatory neurons in pop A
NB = 2 * order  # number of excitatory neurons in pop B
NI = 1 * order  # number of inhibitory neurons
N_neurons = NA + NB + NI   # number of neurons in total
N_rec = order * (2+2+1)  # record from all neurons

tau_syn = [parameters['tau_syn_noise'][0],parameters['tau_syn_AMPA'][0], parameters['tau_syn_NMDA'][0], parameters['tau_syn_GABA'][0]]  # [ms]

exc_neuron_params = {
    "E_L": parameters['V_membrane'][0],
    "V_th": parameters['V_threshold'][0],
    "V_reset": parameters['V_reset'][0],
    "C_m": parameters['C_m_ex'][0],
    "tau_m": parameters['tau_m_ex'][0],
    "t_ref": parameters['t_ref_ex'][0], 
    "tau_syn": tau_syn
}
inh_neuron_params = {
    "E_L": parameters['V_membrane'][0],
    "V_th": parameters['V_threshold'][0],
    "V_reset": parameters['V_reset'][0],
    "C_m": parameters['C_m_in'][0],
    "tau_m": parameters['tau_m_in'][0],
    "t_ref": parameters['t_ref_in'][0], 
    "tau_syn": tau_syn
}

nest.SetDefaults("iaf_psc_exp_multisynapse", exc_neuron_params)
pop_A = nest.Create("iaf_psc_exp_multisynapse", NA)
pop_B = nest.Create("iaf_psc_exp_multisynapse", NB)

nest.SetDefaults("iaf_psc_exp_multisynapse", inh_neuron_params)
pop_inh = nest.Create("iaf_psc_exp_multisynapse", NI)

#'''
#'''**********************************************************************************

J = parameters['J'][0]  # mV -> this means that it takes 200 simultaneous events to drive the spiking activity 

J_unit_noise = ComputePSPNorm(parameters['tau_m_ex'][0], parameters['C_m_ex'][0], parameters['tau_syn_noise'][0])
J_norm_noise = J / J_unit_noise 

J_unit_AMPA = ComputePSPNorm(parameters['tau_m_ex'][0], parameters['C_m_ex'][0], parameters['tau_syn_AMPA'][0])
J_norm_AMPA = J / J_unit_AMPA 

J_norm_NMDA = 0.05  # the weight for the NMDA is set at 0.05, cannot compute J_unit_NMDA since tau_syn_NMDA is greater then tau_m_ex

J_unit_GABA = ComputePSPNorm(parameters['tau_m_in'][0], parameters['C_m_in'][0], parameters['tau_syn_GABA'][0])
J_norm_GABA = J / J_unit_GABA

print("noise: %f" % J_norm_noise)
print("AMPA: %f" % J_norm_AMPA)
print("NMDA: %f" % J_norm_NMDA)  
print("GABA: %f" % J_norm_GABA)

#'''
#'''**********************************************************************************

nu_th_noise_ex = (numpy.abs(parameters['V_threshold'][0]) * parameters['C_m_ex'][0]) / (J_norm_noise * numpy.exp(1) * parameters['tau_m_ex'][0] * parameters['tau_syn_noise'][0])
nu_ex = parameters['eta_ex'][0] * nu_th_noise_ex
p_rate_ex = 1000.0 * nu_ex

p_rate_in = p_rate_ex/ parameters['ratio_noise_rate'][0]    # the rate for the noise entering in the inhibitory population is reduced 
print("p_rate_in: %f" % p_rate_in)

nu_th_noise_in = (numpy.abs(parameters['V_threshold'][0]) * parameters['C_m_in'][0]) / (J_norm_noise * numpy.exp(1) * parameters['tau_m_in'][0] * parameters['tau_syn_noise'][0])
nu_in = parameters['eta_in'][0] * nu_th_noise_in
p_rate_in = 1000.0 * nu_in

print("p_rate_ex: %f" % p_rate_ex)
print("p_rate_in: %f" % p_rate_in)

p_rate_stimulus=  p_rate_ex / parameters['ratio_stim_rate'][0]   #rate for the input Poisson generator to popA (scaled with respect to the noise)
p_rate_stimulus_A = (p_rate_stimulus) * (1 + coherence)
p_rate_stimulus_B = (p_rate_stimulus) * (1 - coherence)

print("p_rate_stimulus: %f" % p_rate_stimulus)

#'''
#'''**********************************************************************************

nest.SetDefaults("poisson_generator", {"rate": p_rate_ex})    #poisson generator for the noise in input to popA and popB
PG_noise_to_ex = nest.Create("poisson_generator")

nest.SetDefaults("poisson_generator", {"rate": p_rate_in})   #poisson generator for the noise in input to popinh
PG_noise_to_inh = nest.Create("poisson_generator")


nest.SetDefaults("poisson_generator", {"rate": p_rate_stimulus_A, "origin": 0.0, "start":parameters['start_stim'][0], "stop":parameters['end_stim'][0]} )   #poisson generator for the input to popA
PG_input_NMDA_A = nest.Create("poisson_generator")
nest.SetDefaults("poisson_generator", {"rate": p_rate_stimulus_A, "origin": 0.0, "start":parameters['start_stim'][0], "stop":parameters['end_stim'][0]} )   #poisson generator for the input to popA
PG_input_AMPA_A = nest.Create("poisson_generator")

nest.SetDefaults("poisson_generator", {"rate": p_rate_stimulus_B, "origin": 0.0, "start":parameters['start_stim'][0], "stop":parameters['end_stim'][0]} )   #poisson generator for the input to popB
PG_input_NMDA_B = nest.Create("poisson_generator")
nest.SetDefaults("poisson_generator", {"rate": p_rate_stimulus_B, "origin": 0.0, "start":parameters['start_stim'][0], "stop":parameters['end_stim'][0]} )   #poisson generator for the input to popB
PG_input_AMPA_B = nest.Create("poisson_generator")

#'''
#'''**********************************************************************************

spikes_a = nest.Create("spike_detector")
spikes_b = nest.Create("spike_detector")
spikes_i = nest.Create("spike_detector")

nest.SetStatus(spikes_a, [{"label": "Excitatory population A",
                          "withtime": True,
                          "withgid": True,
                          "to_file": True}])

nest.SetStatus(spikes_b, [{"label": "Excitatory population B",
                          "withtime": True,
                          "withgid": True,
                          "to_file": True}])

nest.SetStatus(spikes_i, [{"label": "Inhibitory population ",
                          "withtime": True,
                          "withgid": True,
                          "to_file": True}])

#'''
#'''**********************************************************************************

# Definition of synapses
nest.CopyModel("static_synapse", "noise_syn",
               {"weight": J_norm_noise, "delay": parameters['delay_noise'][0]})
noise_syn = {"model": "noise_syn",
                 "receptor_type": 1}

nest.CopyModel("static_synapse", "excitatory_AMPA_input",
               {"weight": J_norm_AMPA, "delay": parameters['delay_AMPA'][0]})
AMPA_input_syn = {"model": "excitatory_AMPA_input",
                 "receptor_type": 2} 
nest.CopyModel("static_synapse", "excitatory_NMDA_input",
               {"weight": J_norm_NMDA, "delay": parameters['delay_NMDA'][0]})
NMDA_input_syn = {"model": "excitatory_NMDA_input",
                 "receptor_type": 3}                              

nest.CopyModel("static_synapse", "excitatory_AMPA_AB_BA",
               {"weight": J_norm_AMPA*parameters['w_minus'][0], "delay": parameters['delay_AMPA'][0]})
AMPA_AB_BA_syn = {"model": "excitatory_AMPA_AB_BA",
                 "receptor_type": 2}
nest.CopyModel("static_synapse", "excitatory_NMDA_AB_BA",
               {"weight": J_norm_NMDA*parameters['w_minus'][0], "delay": parameters['delay_NMDA'][0]})
NMDA_AB_BA_syn = {"model": "excitatory_NMDA_AB_BA",
                 "receptor_type": 3}               

nest.CopyModel("static_synapse", "excitatory_AMPA_AI_BI",
              {"weight": J_norm_AMPA*parameters['w_plus'][0], "delay": parameters['delay_AMPA'][0]})
AMPA_AI_BI_syn = {"model": "excitatory_AMPA_AI_BI",
                 "receptor_type": 2}                  
nest.CopyModel("static_synapse", "excitatory_NMDA_AI_BI",
               {"weight": J_norm_NMDA*parameters['w_plus'][0], "delay": parameters['delay_NMDA'][0]})
NMDA_AI_BI_syn = {"model": "excitatory_NMDA_AI_BI",
                 "receptor_type": 3}  

nest.CopyModel("static_synapse", "inhibitory_IA_IB",
               {"weight": -J_norm_GABA*parameters['w_plus'][0], "delay": parameters['delay_GABA'][0]})
GABA_IA_IB_syn = {"model": "inhibitory_IA_IB",
                 "receptor_type": 4} 

nest.CopyModel("static_synapse", "excitatory_AMPA_recurrent",
              {"weight": J_norm_AMPA, "delay": parameters['delay_AMPA'][0]})
AMPA_recurrent_syn = {"model": "excitatory_AMPA_recurrent",
                 "receptor_type": 2}    
nest.CopyModel("static_synapse", "excitatory_NMDA_recurrent",
              {"weight": J_norm_NMDA*parameters['w_plus'][0]*2.5, "delay": parameters['delay_NMDA'][0]})
NMDA_recurrent_syn = {"model": "excitatory_NMDA_recurrent",
                 "receptor_type": 3} 
nest.CopyModel("static_synapse", "inhibitory_recurrent",
               {"weight": -J_norm_GABA, "delay": parameters['delay_GABA'][0]})
GABA_recurrent_syn = {"model": "inhibitory_recurrent",
                 "receptor_type": 4} 

#'''
#'''**********************************************************************************

#Connecting device

nest.Connect(PG_noise_to_ex, pop_A, syn_spec=noise_syn)
nest.Connect(PG_noise_to_ex, pop_B, syn_spec=noise_syn)
nest.Connect(PG_noise_to_inh, pop_inh, syn_spec=noise_syn)

nest.Connect(PG_input_AMPA_A, pop_A, syn_spec=AMPA_input_syn)
nest.Connect(PG_input_NMDA_A, pop_A, syn_spec=NMDA_input_syn)

nest.Connect(PG_input_AMPA_B, pop_B, syn_spec=AMPA_input_syn)
nest.Connect(PG_input_NMDA_B, pop_B, syn_spec=NMDA_input_syn)

#Recording devices
nest.Connect(pop_A, spikes_a)
nest.Connect(pop_B, spikes_b)
nest.Connect(pop_inh, spikes_i)

#'''
#'''**********************************************************************************

#Connecting populations
conn_params_ex_AB_BA = {'rule': 'pairwise_bernoulli', 'p':parameters['epsilon_ex_AB_BA'][0]}
conn_params_ex_reccurent = {'rule': 'pairwise_bernoulli', 'p':parameters['epsilon_ex_reccurent'][0]}
conn_params_ex_AI_BI = {'rule': 'pairwise_bernoulli', 'p':parameters['epsilon_ex_AI_BI'][0]}
conn_params_in_IA_IB = {'rule': 'pairwise_bernoulli', 'p':parameters['epsilon_in_IA_IB'][0]}
conn_params_in_recurrent = {'rule': 'pairwise_bernoulli', 'p':parameters['epsilon_in_recurrent'][0]}



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

endbuild = time.time() # Storage of the time point after the buildup of the network in a variable.
print("Simulating")
nest.Simulate(simtime)
endsimulate = time.time()
# Reading out the total number of spikes received from the spike recorder connected to 
# the excitatory population and the inhibitory population.
events_ex_a = nest.GetStatus(spikes_a, "n_events")[0]
events_ex_b = nest.GetStatus(spikes_b, "n_events")[0]
events_in = nest.GetStatus(spikes_i, "n_events")[0]

# Calculation of the average firing rate of the excitatory and the inhibitory neurons 
# by dividing the total number of recorded spikes by the number of neurons recorded from and the simulation time. 
# The multiplication by 1000.0 converts the unit 1/ms to 1/s=Hz.
rate_ex_a = events_ex_a / simtime * 1000.0 / NA
rate_ex_b = events_ex_b / simtime * 1000.0 / NB
rate_in = events_in / simtime * 1000.0 / NI

# Establishing the time it took to build and simulate the network by taking the difference of the pre-defined time variables.
build_time = endbuild - startbuild
sim_time = endsimulate - endbuild
# Printing the network properties, firing rates and building times.
print("Excitatory rate A  : %.2f Hz" % rate_ex_a)
print("Excitatory rate B  : %.2f Hz" % rate_ex_b)
print("Inhibitory rate   : %.2f Hz" % rate_in)
print("Building time     : %.2f s" % build_time)
print("Simulation time   : %.2f s" % sim_time)

print("Excitatory events A  : %.2f" % events_ex_a)
print("Excitatory events B  : %.2f" % events_ex_b)
print("Inhibitory events : %.2f" % events_in)

saving_dir = 'results/'+notes+'/'
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)
current_path = os.getcwd()+'/'
shutil.copy(current_path+'parameters.csv',current_path+saving_dir+'parameters.csv')

nest.raster_plot.from_device(spikes_a, hist=True)
plt.savefig(saving_dir+'PopA_'+notes+'png')
nest.raster_plot.from_device(spikes_b, hist=True)
plt.savefig(saving_dir+'PopB_'+notes+'png')
nest.raster_plot.from_device(spikes_i, hist=True)
#plt.show()

events_a = nest.GetStatus(spikes_a,"events")
df = pd.DataFrame(events_a)
senders = pd.DataFrame(df.senders.array)
times = pd.DataFrame(df.times.array)
senders = senders.T
times = times.T
PopA = pd.concat([senders,times],axis = 1)
PopA.columns =['Senders', 'Time']
PopA.to_csv(saving_dir+'PopA_'+notes+'.csv', index = False, float_format = '%.2f')

events_B = nest.GetStatus(spikes_b,"events")
df_B = pd.DataFrame(events_B)
senders_B = pd.DataFrame(df_B.senders.array)
times_B = pd.DataFrame(df_B.times.array)
senders_B = senders_B.T
times_B = times_B.T
PopB = pd.concat([senders_B,times_B],axis = 1)
PopB.columns =['Senders', 'Time']
PopB.to_csv(saving_dir+'PopB_'+notes+'.csv', index = False, float_format = '%.2f')

events_i = nest.GetStatus(spikes_i,"events")
df_i = pd.DataFrame(events_i)
senders_i = pd.DataFrame(df_i.senders.array)
times_i = pd.DataFrame(df_i.times.array)
senders_i = senders_i.T
times_i = times_i.T
PopI = pd.concat([senders_i,times_i],axis = 1)
PopI.columns =['Senders', 'Time']
PopI.to_csv(saving_dir+'PopI_'+notes+'.csv', index = False, float_format = '%.2f')
#'''