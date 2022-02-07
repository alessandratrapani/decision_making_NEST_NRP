import numpy
import time
import matplotlib.pyplot as plt
import os
import sys 
import nest
import nest.raster_plot
import pandas as pd

nest.ResetKernel()
startbuild = time.time()

def LambertWm1(x):
    return nest.ll_api.sli_func('LambertWm1', float(x))

def ComputePSPNorm(tau_mem, C_mem, tau_syn):
    a = (tau_mem / tau_syn)
    b = (1.0 / tau_syn -1.0 / tau_mem)
    t_max = 1.0 / b * (-LambertWm1(-numpy.exp(-1.0/a)/a) - 1.0 / a)
    return (numpy.exp(1.0) / (tau_syn * (C_mem * b) * 
            ((numpy.exp( -t_max / tau_mem) - numpy.exp(-t_max / tau_syn)) / b - 
            t_max * numpy.exp(-t_max / tau_syn))))

dt = 0.1   # the resolution in ms
simtime = 3000.0  # Simulation time in ms
order = 200
NA = 2 * order  # number of excitatory neurons in pop A
NB = 2 * order  # number of excitatory neurons in pop B
NI = 1 * order  # number of inhibitory neurons
N_neurons = NA + NB + NI   # number of neurons in total
N_rec = order * (2+2+1)  # record from all neurons

parameters = pd.read_csv('parameters.csv')

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

# external rate relative to threshold rate
eta_ex = 2.0 
eta_in = eta_ex

nu_th_noise_ex = (numpy.abs(parameters['V_threshold'][0]) * parameters['C_m_ex'][0]) / (J_norm_noise * numpy.exp(1) * parameters['tau_m_ex'][0] * parameters['tau_syn_noise'][0])
nu_ex = parameters['eta_ex'][0] * nu_th_noise_ex
p_rate_ex = 1000.0 * nu_ex

p_rate = p_rate_ex/ parameters['ratio_noise_rate'][0]    # the rate for the noise entering in the inhibitory population is reduced 

nu_th_noise_in = (numpy.abs(parameters['V_threshold'][0]) * parameters['C_m_in'][0]) / (J_norm_noise * numpy.exp(1) * parameters['tau_m_in'][0] * parameters['tau_syn_noise'][0])
nu_in = parameters['eta_in'][0] * nu_th_noise_in
p_rate_in = 1000.0 * nu_in

print("p_rate_ex: %f" % p_rate_ex)
print("p_rate: %f" % p_rate)
print("p_rate_in: %f" % p_rate_in)

nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
