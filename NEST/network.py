import numpy
import time
import matplotlib.pyplot as plt
import os
import sys 
import nest
import nest.raster_plot

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

tau_m_ex = 20.0  # time constant of membrane potential in ms
tau_m_in = 10.0
C_m_ex = 500.
C_m_in = 200.
theta = -55.0  # membrane threshold potential in mV
t_ref_ex = 2.0
t_ref_in = 1.0
V_membrane = -70.0  # mV 
V_threshold = -50.0  # mV
V_reset = -55.0  # mV 

nr_ports = 4  # number of receptor types (noise-related, AMPA, NMDA, GABA)  
tau_syn_noise = 5.
tau_syn_AMPA = 2.
tau_syn_NMDA = 100.
tau_syn_GABA = 5.
tau_syn = [tau_syn_noise, tau_syn_AMPA, tau_syn_NMDA, tau_syn_GABA]  # [ms]

exc_neuron_params = {
    "E_L": V_membrane,
    "V_th": V_threshold,
    "V_reset": V_reset,
    "C_m": C_m_ex,
    "tau_m": tau_m_ex,
    "t_ref": t_ref_ex, 
    "tau_syn": tau_syn
}
_neuron_params = {
    "E_L": V_membrane,
    "V_th": V_threshold,
    "V_reset": V_reset,
    "C_m": C_m_in,
    "tau_m": tau_m_in,
    "t_ref": t_ref_in, 
    "tau_syn": tau_syn
}

J = 0.1  # mV -> this means that it takes 200 simultaneous events to drive the spiking activity 

J_unit_noise = ComputePSPNorm(tau_m_ex, C_m_ex, tau_syn_noise)
J_norm_noise = J / J_unit_noise 

J_unit_AMPA = ComputePSPNorm(tau_m_ex, C_m_ex, tau_syn_AMPA)
J_norm_AMPA = J / J_unit_AMPA 

J_norm_NMDA = 0.05  # the weight for the NMDA is set at 0.05, cannot compute J_unit_NMDA since tau_syn_NMDA is greater then tau_m_ex

J_unit_GABA = ComputePSPNorm(tau_m_in, C_m_in, tau_syn_GABA)
J_norm_GABA = J / J_unit_GABA

print("noise: %f" % J_norm_noise)
print("AMPA: %f" % J_norm_AMPA)
print("NMDA: %f" % J_norm_NMDA)  
print("GABA: %f" % J_norm_GABA)

# external rate relative to threshold rate
eta_ex = 2.0 
eta_in = eta_ex

nu_th_noise_ex = (numpy.abs(V_threshold) * C_m_ex) / (J_norm_noise * numpy.exp(1) * tau_m_ex * tau_syn_noise)
nu_ex = eta_ex * nu_th_noise_ex
p_rate_ex = 1000.0 * nu_ex

p_rate = p_rate_ex/ 1.3   # the rate for the noise entering in the inhibitory population is reduced 

nu_th_noise_in = (numpy.abs(V_threshold) * C_m_in) / (J_norm_noise * numpy.exp(1) * tau_m_in * tau_syn_noise)
nu_in = eta_in * nu_th_noise_in
p_rate_in = 1000.0 * nu_in

print("p_rate_ex: %f" % p_rate_ex)
print("p_rate: %f" % p_rate)
print("p_rate_in: %f" % p_rate_in)

nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})

