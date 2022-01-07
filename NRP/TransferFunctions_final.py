########################################################################################
############################ NRP TRANSFER FUNCTIONS ####################################
########################################################################################

################### iCube head motion
from __future__ import division

# Imported Python Transfer Function
import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import std_msgs.msg
@nrp.MapSpikeSink("PopA", nrp.brain.actuators[slice(4,44,1)], nrp.population_rate)
@nrp.MapSpikeSink("PopB", nrp.brain.actuators[slice(44,84,1)], nrp.population_rate)
@nrp.Neuron2Robot(Topic('/icub_hbp_ros_0/neck_yaw/pos', std_msgs.msg.Float64))
def head_twist(t, PopA, PopB):
    from past.utils import old_div

    frequency_right=PopA.rate
    frequency_left=PopB.rate
    data=(50.0 * (frequency_right - frequency_left))
    if(data<50.0*200.0 and data>(-50.0*200.0)):
        data = 0
    # if abs(data)>0.3:
    #             sign=old_div(data,(abs(data)))
    #             data=0.3*sign
    #if rate maggiore di threshold -->
    return std_msgs.msg.Float64(data)
#

#################### iCub eye detection(new)
@nrp.MapRobotSubscriber("camera_left", Topic('/icub_hbp_ros_0/icub_model/right_eye_camera/image_raw', sensor_msgs.msg.Image))
@nrp.MapSpikeSource("red_eye", nrp.brain.sensors[slice(0,2,1)], nrp.poisson)
@nrp.MapSpikeSource("blue_eye", nrp.brain.sensors[slice(2,4,1)], nrp.poisson)
@nrp.Robot2Neuron()

def eye_sensor_transmit_right (t, camera_left, red_eye, blue_eye):
    count_red = 0
    count_blue = 0
    image_results_left = hbp_nrp_cle.tf_framework.tf_lib.get_color_values(image=camera_left.value)
    red_array = image_results_left.right_red + image_results_left.left_red
    blue_array = image_results_left.right_blue + image_results_left.left_blue
    #clientLogger.info('Red array', red_array)
    for x in red_array:
        if(x>=0.8):
            count_red = count_red + 1
    
    for y in blue_array:
        if(y>=0.8):
            count_blue = count_blue + 1
    
    clientLogger.info('Count_red', count_red)
    clientLogger.info('Count_blue', count_blue)

    rate_tot = 335.0255
    rate_red = float(count_red)
    rate_blue = float(count_blue)
    if((rate_red + rate_blue) != 0):
        k = rate_red/(rate_red+rate_blue)

        #estrarre pixels sopra threshold e sommare in red_eye.rate
        red_eye.rate = float(k*rate_tot)
        blue_eye.rate = float((1.0-k)*rate_tot)
#

##################### Spike recorder
@nrp.NeuronMonitor(nrp.brain.recorders[slice(0, 100, 1)], nrp.spike_recorder)
def all_neurons_spike_monitor (t):
    return True
#
