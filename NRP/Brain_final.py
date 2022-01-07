from __future__ import division
from builtins import str
# pragma: no cover

import nest
import logging
import numpy
from hbp_nrp_cle.brainsim import simulator as sim
from hbp_nrp_excontrol.logs import clientLogger
import pyNN

__author__ = 'Group 2'

logger = logging.getLogger(__name__)

# Helper functions to automatically tune the network weights so to have near-threshold dynamics
def LambertWm1(x):
    return nest.ll_api.sli_func('LambertWm1', float(x))

def ComputePSPNorm(tau_mem, C_mem, tau_syn, is_NMDA=False):
    a = (tau_mem / tau_syn)
    if is_NMDA:
        a *= 5.02
    b = (1.0 / tau_syn -1.0 / tau_mem)
    t_max = 1.0 / b * (-LambertWm1(-numpy.exp(-1.0/a)/a) - 1.0 / a)
    return (numpy.exp(1.0) / (tau_syn * (C_mem * b) * 
            ((numpy.exp( -t_max / tau_mem) - numpy.exp(-t_max / tau_syn)) / b - 
            t_max * numpy.exp(-t_max / tau_syn))))


# --------------------------------
# Main function: 
# Returns the spiking circuit to be connected to the robotic agent
# --------------------------------
def create_brain():
    #dt = 1.0    # the resolution in ms
    simtime = 3000.0  # Simulation time in ms
    delay = 0.5    # Synaptic delay in ms
    
    eta = 2.0  # External rate relative to threshold rate
    epsilon = 0.4  # Connection probability between populations
    epsilon_same = 0.4 # connection probability between neurons of a same population
    epsilon_opposite = 0.3 # connection probability between neurons of a different population
    epsilon_inh_to_pop = 0.4 # connection probability between inhibitory interneurons and the 2 populations and viceversa
    epsilon_inh_same = 0.4 # connection probability of recurrent connections for inhibitory interneurons
    epsilon_input = 1.0 # connection probability of the external input (input to all neurons in this case)

    order = 20 
    NA = 2 * order  # Number of excitatory neurons in pop A
    NB = 2 * order  # Number of excitatory neurons in pop A
    NI = 1 * order  # Number of inhibitory neurons
    N_neurons = NA + NB + NI   # Number of neurons in total

    CE = int(epsilon_same*(NA) + epsilon_opposite*(NB))  # Number of excitatory synapses per neuron 
    CI = int(epsilon_inh_to_pop * NI)  # Number of inhibitory synapses per neuron
    C_tot = int(CI + CE)      # Total number of synapses per neuron

    tauMem = 20.0  # Time constant of membrane potential in ms
    theta = -55.0  # Membrane threshold potential in mV

    nr_ports = 4  # Number of receptor types (noise-related, AMPA, NMDA, GABA)
    tau_syn = [5., 2., 100., 5.]  # Exponential time constant of post-synaptic current (PSC) for each receptor [ms]

    # Population-independent constants
    V_membrane = -70.0  # [mV] 
    V_threshold = -50.0  # [mV]
    V_reset = -55.0  # [mV]

    # Population-dependent constants
    # Excitatory
    C_m_ex = 500.0  # [pF]  
    # Inhibitory
    C_m_in = 200.0  # [pF]
    # Excitatory
    t_ref_ex = 2.0 
    # Inhibitory
    t_ref_in = 1.0
    # Excitatory
    tau_m_ex = 20.0
    # Inhibitory
    tau_m_in = 10.0

    # Storing the neurons' parameters in a dictionary to be assigned to the models later
    # Excitatory
    exc_neuron_params = {
        "E_L": V_membrane,
        "V_th": V_threshold,
        "V_reset": V_reset,
        "C_m": C_m_ex,  
        "tau_m": tau_m_ex,
        "t_ref": t_ref_ex, 
        "tau_syn": tau_syn
    }
    # Inhibitory 
    inh_neuron_params = {
        "E_L": V_membrane,
        "V_th": V_threshold,
        "V_reset": V_reset,
        "C_m": C_m_in, 
        "tau_m": tau_m_in,
        "t_ref": t_ref_in,
        "tau_syn": tau_syn 
    } 

    # Automatic weight tuning 
    J = 0.04  # Post-synaptic potential (PSP) induced by a single spike  [mV]  
    # Noise weights
    J_unit_noise = ComputePSPNorm(tau_m_ex, C_m_ex, tau_syn[0])
    J_norm_noise = J / J_unit_noise 
    # AMPA weights
    J_unit_AMPA = ComputePSPNorm(tau_m_ex, C_m_ex, tau_syn[1])
    J_norm_AMPA = J / J_unit_AMPA
    J_norm_AMPA = J_norm_AMPA 
    # NMDA weights
    J_unit_NMDA = ComputePSPNorm(tau_m_ex, C_m_ex, tau_syn[2], is_NMDA = True)
    J_norm_NMDA = J / J_unit_NMDA
    J_unit_NMDA = 0.04  # the weight for the NMDA is set at 0.05, otherwise it would be 0
    # GABA weights
    J_unit_GABA = ComputePSPNorm(tau_m_in, C_m_in, tau_syn[3])
    J_norm_GABA = J / J_unit_GABA

    # Reference code
    # NOTE: setting the rate of the input in a brain_model scripts is useless, 
    # as the NRP only transmits external spike to the network through the transfer functions. 
    # We kept the code nonetheless to remind you of what was done in NEST and make you think of how 
    # to replicate it on the Platform
    nu_th_noise_ex = (numpy.abs(V_threshold) * C_m_ex) / (J_norm_noise * CE * numpy.exp(1) * tau_m_ex * 10.0)
    nu_ex = eta * nu_th_noise_ex
    p_rate = 1000.0 * nu_ex * CE
    p_rate_inh = p_rate / 1.25   # the rate for the noise entering in the inhibitory population is reduced 

    nest.SetDefaults("poisson_generator", {"rate": p_rate})    #poisson generator for the noise in input to popA and popB
    PG_noise = nest.Create("poisson_generator")

    nest.SetDefaults("poisson_generator", {"rate": p_rate_inh})   #poisson generator for the noise in input to popinh
    PG_noise_to_inh = nest.Create("poisson_generator")


    #--------------------------------
    # Creating the nodes
    # --------------------------------
    # Defining the populations models and creating the nodes
    inputs = nest.Create("parrot_neuron", 4)  # Parrot neurons to transmit the inputs to the network, only act as separate buffers (one for each pop)   
    population = nest.Create("iaf_psc_exp_multisynapse", N_neurons)  # Instantiating a general (parameter-less) neuronal population (of current-based I&F) to model all the neurons
    nest.SetStatus(population[0: 4 * order - 1], exc_neuron_params)  # Defining the excitatory subpopulation numerosity and parameters
    nest.SetStatus(population[4 * order : 5 * order - 1], inh_neuron_params)  # Defining the inhibitory subpopulation numerosity and parameters

    # Defining the quantities to be used in the connectivity instructions
    # Subpopulation A and relative input  
    pop_A = population[0:(order * 2 -1)]  
    input_A_AMPA = inputs[0:1]
    input_A_NMDA = inputs[1:2]
   # input_noise_A= inputs[2:3]
    # Subpopulation B and relative input  
    pop_B = population[(order * 2) : order * 4 -1]
    input_B_AMPA = inputs[2:3]
    input_B_NMDA = inputs[3:4]
   # input_noise_B= inputs[5:6]
    # Subpopulation C and relative input  
    pop_inh = population[(order * 4) : (order * 5 -1)]
   # input_noise_inh= inputs[6:7]

    w_minus = 0.85

    # Defining the synaptic connections names, parameters and receptor type to be used later
    nest.CopyModel("static_synapse", "excitatory_AMPA",
               {"weight": J_norm_AMPA, "delay": delay})
    nest.CopyModel("static_synapse", "excitatory_NMDA",
                {"weight": J_norm_NMDA*15.0, "delay": delay})
    nest.CopyModel("static_synapse", "noise_syn",
                {"weight": J_norm_noise, "delay": delay})
    nest.CopyModel("static_synapse", "inhibitory",
                {"weight": -J_norm_GABA, "delay": delay})

    #Definition of new types of synapses considering a hebbian rule for the weights (recurrent connections)
    nest.CopyModel("static_synapse", "excitatory_AMPA_same",
                  {"weight": J_norm_AMPA, "delay": delay})      # the weight for the recurrent AMPA connections is reduced a lot (ramping activity is mainly due to NMDA reverberations)
    nest.CopyModel("static_synapse", "excitatory_NMDA_same",
                   {"weight": J_norm_NMDA*47.5, "delay": delay})   #  the weight for the recurrent NMDA connections is increased (ramping activity is mainly due to NMDA 

    # Definition of new types of synapses considering a hebbian rule for the weights (connections between different populations)
    nest.CopyModel("static_synapse", "excitatory_AMPA_opposite",
                  {"weight": J_norm_AMPA*w_minus, "delay": delay})  # the weight for the AMPA connections to the opposite population is reduced 
    nest.CopyModel("static_synapse", "excitatory_NMDA_opposite",
                   {"weight": J_norm_NMDA*w_minus, "delay": delay})   # the weight for the NMDA connections to the opposite population is reduced  

    # Definition of new types of synapses for the input (to popA only) by considering weights of the exc neurotransmitters or the same weight used for the noise
    nest.CopyModel("static_synapse", "excitatory_NMDA_input",
                   {"weight": J_norm_NMDA*15.0, "delay": delay})  # the weight for the input is increased and it-s delayed by 0.5 x 400 ms, i.e 200ms
    nest.CopyModel("static_synapse", "excitatory_AMPA_input",
                   {"weight": J_norm_AMPA, "delay": delay})  # the weight for the input is increased and it-s delayed by 0.5 x 400 ms, i.e 200ms

    # Definition of new types of synapses for the popinh to popA and popB connections (maybe a higher delay cam be useful to see a decreasing activity of the non stimulated pop)
    nest.CopyModel("static_synapse", "inhibitory_opposite",
                   {"weight": -J_norm_GABA*10.0, "delay": delay})  # the weight for the inhibitory connections to popA and popB is increased
    
    # #DEFAULT NAMES FOR THE SYNAPSES
    noise_syn = {"model": "noise_syn",
                 "receptor_type": 1}
    AMPA_syn = {"model": "excitatory_AMPA",
                    "receptor_type": 2}
    NMDA_syn = {"model": "excitatory_NMDA",
                    "receptor_type": 3}
    GABA_syn = {"model": "inhibitory",
                    "receptor_type": 4}

    #NEW SYNAPSES
    AMPA_syn_same = {"model": "excitatory_AMPA_same",
                         "receptor_type": 2}
    NMDA_syn_same = {"model": "excitatory_NMDA_same",
                         "receptor_type": 3}

    AMPA_syn_opposite = {"model": "excitatory_AMPA_opposite",
                         "receptor_type": 2}
    NMDA_syn_opposite = {"model": "excitatory_NMDA_opposite",
                         "receptor_type": 3}


    AMPA_syn_input = {"model": "excitatory_AMPA_input",
                         "receptor_type": 2}
    NMDA_syn_input = {"model": "excitatory_NMDA_input",
                         "receptor_type": 3}

    GABA_syn_opposite = {"model": "inhibitory_opposite",
                     "receptor_type": 4}
    
    #--------------------------------
    # Connecting the nodes
    # -------------------------------

    # Defining the connectivty strategy to be used
    conn_params_ex_same = {'rule': 'pairwise_bernoulli', 'p': epsilon_same}
    conn_params_ex_opposite = {'rule': 'pairwise_bernoulli', 'p': epsilon_opposite}
    conn_params_ex_inh_to_pop = {'rule': 'pairwise_bernoulli', 'p': epsilon_inh_to_pop}
    conn_params_ex_inh_same = {'rule': 'pairwise_bernoulli', 'p': epsilon_inh_same}
    conn_params_in = {'rule': 'pairwise_bernoulli', 'p': epsilon}
    
    # Connecting the components

    #Noise
    # nest.Connect(input_noise_A, pop_A, 'all_to_all', {"receptor_type": 1})
    # nest.Connect(input_noise_B, pop_B, 'all_to_all', {"receptor_type": 1})
    # nest.Connect(input_noise_inh, pop_inh, 'all_to_all', {"receptor_type": 1})
    nest.Connect(PG_noise, pop_A, syn_spec=noise_syn)
    nest.Connect(PG_noise, pop_B, syn_spec=noise_syn)
    nest.Connect(PG_noise_to_inh, pop_inh, syn_spec=noise_syn)
    

    #Input
    nest.Connect(input_A_NMDA, pop_A, 'all_to_all', {"receptor_type": 3})
    nest.Connect(input_A_AMPA, pop_A, 'all_to_all', {"receptor_type": 2})
    nest.Connect(input_B_NMDA, pop_B, 'all_to_all', {"receptor_type": 3})
    nest.Connect(input_B_AMPA, pop_B, 'all_to_all', {"receptor_type": 2})

    # Population A

    # Recurrent
    nest.Connect(pop_A, pop_A, conn_params_ex_same, AMPA_syn_same)
    nest.Connect(pop_A, pop_A, conn_params_ex_same, NMDA_syn_same)
    # To pop B
    nest.Connect(pop_A, pop_B, conn_params_ex_opposite, AMPA_syn_opposite)
    nest.Connect(pop_A, pop_B, conn_params_ex_opposite, NMDA_syn_opposite)
    # To pop inh.
    nest.Connect(pop_A, pop_inh, conn_params_ex_inh_to_pop, AMPA_syn)
    nest.Connect(pop_A, pop_inh, conn_params_ex_inh_to_pop, NMDA_syn)

    # Population B

    # Recurrent
    nest.Connect(pop_B, pop_B, conn_params_ex_same, AMPA_syn_same)
    nest.Connect(pop_B, pop_B, conn_params_ex_same, NMDA_syn_same)
    # To pop A
    nest.Connect(pop_B, pop_A, conn_params_ex_opposite, AMPA_syn_opposite)
    nest.Connect(pop_B, pop_A, conn_params_ex_opposite, NMDA_syn_opposite)
    # To pop inh.
    nest.Connect(pop_B, pop_inh, conn_params_ex_inh_to_pop, AMPA_syn)
    nest.Connect(pop_B, pop_inh, conn_params_ex_inh_to_pop, NMDA_syn)

    # Population inh 

    # Recurrent
    nest.Connect(pop_inh, pop_inh, conn_params_ex_inh_same, GABA_syn)
    # To pop A
    nest.Connect(pop_inh, pop_A, conn_params_ex_inh_to_pop, GABA_syn_opposite)
    # To pop B
    nest.Connect(pop_inh, pop_B, conn_params_ex_inh_to_pop, GABA_syn_opposite)


    #--------------------------------
    # End of function
    # -------------------------------
    # Finally, return the network composed by the sum of the two distinct node types (populations), 
    # the "input cells" and neuronal cells
    nodes = inputs + population
    return nodes

circuit = create_brain()