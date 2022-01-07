'''
This is the State Machine implementation for the simulation of a winner-takes-all 
mechanism modelled with a neural network implemented in the iCub brain.
On the scene some boxes will appear and their color will change. Depending on the 
color detected by the iCub the head is rotated in the right or left direction.

-------------- DO NOT COPY THIS COMMENT WHEN IMPORTING IN NRP ---------------------
'''

#!/usr/bin/env python
"""
A state-machine that deletes, spawns and moves objects in the 3D scenes.
"""
from __future__ import division

from past.utils import old_div
__author__ = 'Group 2'

import math
import time
import rospy
import smach
from smach import StateMachine
from smach import CBState
from hbp_nrp_excontrol.nrp_states import WaitToClockState, RobotPoseMonitorState, \
    SetMaterialColorServiceState, ClockDelayState, SpawnSphere, SpawnBox, DestroyModel, SetModelPose
from hbp_nrp_excontrol.logs import clientLogger
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState


@smach.cb_interface(input_keys=[''], output_keys=[''], outcomes=['succeeded'])
def init(userdata):
    time.sleep(1)
    clientLogger.advertise('Computational neuroscience - Project workshop')
    time.sleep(5)
    clientLogger.advertise('Developed by')
    time.sleep(2)
    clientLogger.advertise('Camporeale G., Chiosso S., Colombo L., DAndrea M., Gallinea S.')
    time.sleep(5)
    return 'succeeded'


set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)


def move_object_cb(name, pointPosition, qOrientation=Quaternion(0, 0, 0, 0)):
    @smach.cb_interface(input_keys=[], output_keys=[], outcomes=['succeeded', 'aborted'])
    def move_object(userdata):
        msg = ModelState();

        msg.model_name = name
        msg.scale = Vector3(1, 1, 1)
        msg.pose.position = pointPosition
        msg.pose.orientation = qOrientation
        msg.reference_frame = 'world'

        # call service
        response = set_model_state(msg)

        if not response.success:
            return 'aborted'
        return 'succeeded'

    return move_object


def notify_user_cb(msg):
    @smach.cb_interface(input_keys=[], output_keys=[], outcomes=['succeeded'])
    def notify_user(userdata):
        clientLogger.advertise(msg)
        return 'succeeded'

    return notify_user


def moveAlongPath(model, pointFrom, pointTo, stepSize=0.1):
    @smach.cb_interface(input_keys=['counter'],
                        output_keys=['counter'],
                        outcomes=['succeeded', 'ongoing'])
    def move_along(userdata):
        direction = Vector3(pointTo.x - pointFrom.x, pointTo.y - pointFrom.y,
                            pointTo.z - pointFrom.z)
        step = stepSize * userdata.counter
        newPos = Point(pointFrom.x + direction.x * step,
                       pointFrom.y + direction.y * step,
                       pointFrom.z + direction.z * step)
        move_object_cb(model, newPos)(userdata)

        if (userdata.counter < math.fabs(old_div(1, stepSize))):
            userdata.counter = userdata.counter + 1
            return 'ongoing'
        else:
            userdata.counter = 0
            return 'succeeded'

    return move_along

################################################################
####################### STATE MACHINE ##########################
################################################################

FINISHED = 'FINISHED'
ERROR = 'ERROR'
PREEMPTED = 'PREEMPTED'

sm = StateMachine(outcomes=[FINISHED, ERROR, PREEMPTED])
sm.userdata.counter = 0

with sm:
    StateMachine.add('INIT',
                     CBState(init),
                     transitions={'succeeded': 'INITIAL_WAITING'})
   
    StateMachine.add('INITIAL_WAITING',
                     WaitToClockState(1),
                     {'valid': 'INITIAL_WAITING', 
                      'invalid': 'SPAWN_OBJECT0',
                      'preempted': PREEMPTED})

    StateMachine.add('SPAWN_OBJECT0',
                     SpawnBox(model_name="Box0", size = Vector3(1,1,1),
                                 position=Point(-1,0,0.5), gravity_factor=0,
                                ),                            
                     transitions={'succeeded': 'SPAWN_OBJECT1', 
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})
    
    StateMachine.add('SPAWN_OBJECT1',
                     SpawnBox(model_name="Box1", size = Vector3(1,1,1),
                                 position=Point(-1,1,0.5), gravity_factor=0,
                                ),
                     transitions={'succeeded': 'SPAWN_OBJECT2', 
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})
    
    StateMachine.add('SPAWN_OBJECT2',
                 SpawnBox(model_name="Box2", size = Vector3(1,1,1),
                             position=Point(-1,-1,0.5), gravity_factor=0,
                            ),
                 transitions={'succeeded': 'SPAWN_OBJECT3', 
                              'aborted': ERROR,
                              'preempted': PREEMPTED})
    
    StateMachine.add('SPAWN_OBJECT3',
                     SpawnBox(model_name="Box3", size = Vector3(1,1,1),
                                 position=Point(-1,0,1.5), gravity_factor=0,
                                ),
                     transitions={'succeeded': 'SPAWN_OBJECT4', 
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})
    
    StateMachine.add('SPAWN_OBJECT4',
                     SpawnBox(model_name="Box4", size = Vector3(1,1,1),
                                 position=Point(-1,1,1.5), gravity_factor=0,
                                ),
                     transitions={'succeeded': 'SPAWN_OBJECT5', 
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})
    
    StateMachine.add('SPAWN_OBJECT5',
                     SpawnBox(model_name="Box5", size = Vector3(1,1,1),
                                 position=Point(-1,-1,1.5), gravity_factor=0,
                                ),
                     transitions={'succeeded': 'SPAWN_OBJECT6', 
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SPAWN_OBJECT6',
                     SpawnBox(model_name="Box6",size = Vector3(1,1,1),
                                 position=Point(-1,0,2.5), gravity_factor=0,
                                ),
                     transitions={'succeeded': 'SPAWN_OBJECT7', 
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})
     
    StateMachine.add('SPAWN_OBJECT7',
                     SpawnBox(model_name="Box7",size = Vector3(1,1,1),
                                 position=Point(-1,1,2.5), gravity_factor=0,
                                ),
                     transitions={'succeeded': 'SPAWN_OBJECT8', 
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SPAWN_OBJECT8',
                     SpawnBox(model_name="Box8",size = Vector3(1,1,1),
                                 position=Point(-1,-1,2.5), gravity_factor=0,
                                ),
                     transitions={'succeeded': 'START_MSG',     
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('START_MSG',
                     CBState(notify_user_cb('Start of the experiment')),
                     {'succeeded': 'DELAY_ON_MOTION0'})

    StateMachine.add('DELAY_ON_MOTION0',
                     ClockDelayState(2),
                     transitions={'valid': 'DELAY_ON_MOTION0', 
                                  'invalid': 'START_MSG2',    #change here to COHERENCE_MSG to have coherence in input and 
                                  'preempted': PREEMPTED})    #comment the rest of the code up to ########## COHERENCE #############

    StateMachine.add('START_MSG2',
                     CBState(notify_user_cb('The color of the boxes is set to either red or blue and the robot will produce a motor response')),
                     {'succeeded': 'DELAY_ON_MOTION1'})

    StateMachine.add('DELAY_ON_MOTION1',
                         ClockDelayState(3),
                         transitions={'valid': 'DELAY_ON_MOTION1', 
                                      'invalid': 'SWITCH_TO_RED0', #change here to SWITCH_TO_BLUE0 to have blue glow of the spheres
                                      'preempted': PREEMPTED})     #and comment the code up to ############ BLUE GLOW ############

######################## RED GLOW ########################
    
    StateMachine.add('SWITCH_TO_RED0',
                     SetMaterialColorServiceState('Box0',
                                                  'body',
                                                  'visual',
                                                  'Gazebo/RedGlow'),
                     transitions={'succeeded': 'SWITCH_TO_RED1',
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SWITCH_TO_RED1',
                 SetMaterialColorServiceState('Box1',
                                              'body',
                                              'visual',
                                              'Gazebo/RedGlow'),
                 transitions={'succeeded': 'SWITCH_TO_RED2',
                              'aborted': ERROR,
                              'preempted': PREEMPTED})

    StateMachine.add('SWITCH_TO_RED2',
             SetMaterialColorServiceState('Box2',
                                          'body',
                                          'visual',
                                          'Gazebo/RedGlow'),
             transitions={'succeeded': 'SWITCH_TO_RED3',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})
    
    StateMachine.add('SWITCH_TO_RED3',
                     SetMaterialColorServiceState('Box3',
                                                  'body',
                                                  'visual',
                                                  'Gazebo/RedGlow'),
                     transitions={'succeeded': 'SWITCH_TO_RED4',
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SWITCH_TO_RED4',
                 SetMaterialColorServiceState('Box4',
                                              'body',
                                              'visual',
                                              'Gazebo/RedGlow'),
                 transitions={'succeeded': 'SWITCH_TO_RED5',
                              'aborted': ERROR,
                              'preempted': PREEMPTED})

    StateMachine.add('SWITCH_TO_RED5',
             SetMaterialColorServiceState('Box5',
                                          'body',
                                          'visual',
                                          'Gazebo/RedGlow'),
             transitions={'succeeded': 'SWITCH_TO_RED6',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('SWITCH_TO_RED6',
                     SetMaterialColorServiceState('Box6',
                                                  'body',
                                                  'visual',
                                                  'Gazebo/RedGlow'),
                     transitions={'succeeded': 'SWITCH_TO_RED7',
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SWITCH_TO_RED7',
                 SetMaterialColorServiceState('Box7',
                                              'body',
                                              'visual',
                                              'Gazebo/RedGlow'),
                 transitions={'succeeded': 'SWITCH_TO_RED8',
                              'aborted': ERROR,
                              'preempted': PREEMPTED})

    StateMachine.add('SWITCH_TO_RED8',
             SetMaterialColorServiceState('Box8',
                                          'body',
                                          'visual',
                                          'Gazebo/RedGlow'),
             transitions={'succeeded': 'WAIT',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('WAIT',
                    WaitToClockState(20),
                     {'valid': 'WAIT', 'invalid': 'END_MSG',
                      'preempted': PREEMPTED})

    StateMachine.add('END_MSG',
                     CBState(notify_user_cb('The color of the boxes is set back to WHITE to see neuronal persistent activity')),
                     {'succeeded': 'DELAY_ON_END'})

    StateMachine.add('DELAY_ON_END',
                         ClockDelayState(3),
                         transitions={'valid': 'DELAY_ON_END', 
                                      'invalid': 'SET_GRAY0', 
                                      'preempted': PREEMPTED})

    StateMachine.add('SET_GRAY0',
            SetMaterialColorServiceState('Box0',
                                          'body',
                                          'visual',
                                          'Gazebo/WhiteGlow'),
             transitions={'succeeded': 'SET_GRAY1',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('SET_GRAY1',
            SetMaterialColorServiceState('Box1',
                                          'body',
                                          'visual',
                                          'Gazebo/WhiteGlow'),
             transitions={'succeeded': 'SET_GRAY2',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('SET_GRAY2',
            SetMaterialColorServiceState('Box2',
                                          'body',
                                          'visual',
                                          'Gazebo/WhiteGlow'),
             transitions={'succeeded': 'SET_GRAY3',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('SET_GRAY3',
            SetMaterialColorServiceState('Box3',
                                          'body',
                                          'visual',
                                          'Gazebo/WhiteGlow'),
             transitions={'succeeded': 'SET_GRAY4',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('SET_GRAY4',
            SetMaterialColorServiceState('Box4',
                                          'body',
                                          'visual',
                                          'Gazebo/WhiteGlow'),
             transitions={'succeeded': 'SET_GRAY5',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('SET_GRAY5',
            SetMaterialColorServiceState('Box5',
                                          'body',
                                          'visual',
                                          'Gazebo/WhiteGlow'),
             transitions={'succeeded': 'SET_GRAY6',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('SET_GRAY6',
            SetMaterialColorServiceState('Box6',
                                          'body',
                                          'visual',
                                          'Gazebo/WhiteGlow'),
             transitions={'succeeded': 'SET_GRAY7',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('SET_GRAY7',
            SetMaterialColorServiceState('Box7',
                                          'body',
                                          'visual',
                                          'Gazebo/WhiteGlow'),
             transitions={'succeeded': 'SET_GRAY8',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('SET_GRAY8',
            SetMaterialColorServiceState('Box8',
                                          'body',
                                          'visual',
                                          'Gazebo/WhiteGlow'),
             transitions={'succeeded': 'END_MSG2',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('END_MSG2',
                     CBState(notify_user_cb('It can be seen that the neuronal activity holds for more or less 2s. Then random firing restarts')),
                     {'succeeded': 'DELAY_ON_END2'})

    StateMachine.add('DELAY_ON_END2',
                         ClockDelayState(15),
                         transitions={'valid': 'DELAY_ON_END', 
                                      'invalid': 'WAIT2', 
                                      'preempted': PREEMPTED})

    StateMachine.add('WAIT2',
                    WaitToClockState(10),
                     {'valid': 'WAIT2', 'invalid': FINISHED,
                      'preempted': PREEMPTED})

############################## BLUE GLOW ############################
# To change the color of the spheres to blue comment the second part of the
# state machine (where indicated) and uncomment this one below.

    StateMachine.add('SWITCH_TO_BLUE0',
                     SetMaterialColorServiceState('Box0',
                                                  'body',
                                                  'visual',
                                                  'Gazebo/BlueGlow'),
                     transitions={'succeeded': 'SWITCH_TO_BLUE1',
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SWITCH_TO_BLUE1',
                 SetMaterialColorServiceState('Box1',
                                              'body',
                                              'visual',
                                              'Gazebo/BlueGlow'),
                 transitions={'succeeded': 'SWITCH_TO_BLUE2',
                              'aborted': ERROR,
                              'preempted': PREEMPTED})

    StateMachine.add('SWITCH_TO_BLUE2',
             SetMaterialColorServiceState('Box2',
                                          'body',
                                          'visual',
                                          'Gazebo/BlueGlow'),
             transitions={'succeeded': 'SWITCH_TO_BLUE3',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})
    
    StateMachine.add('SWITCH_TO_BLUE3',
                     SetMaterialColorServiceState('Box3',
                                                  'body',
                                                  'visual',
                                                  'Gazebo/BlueGlow'),
                     transitions={'succeeded': 'SWITCH_TO_BLUE4',
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SWITCH_TO_BLUE4',
                 SetMaterialColorServiceState('Box4',
                                              'body',
                                              'visual',
                                              'Gazebo/BlueGlow'),
                 transitions={'succeeded': 'SWITCH_TO_BLUE5',
                              'aborted': ERROR,
                              'preempted': PREEMPTED})

    StateMachine.add('SWITCH_TO_BLUE5',
             SetMaterialColorServiceState('Box5',
                                          'body',
                                          'visual',
                                          'Gazebo/BlueGlow'),
             transitions={'succeeded': 'SWITCH_TO_BLUE6',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('SWITCH_TO_BLUE6',
                     SetMaterialColorServiceState('Box6',
                                                  'body',
                                                  'visual',
                                                  'Gazebo/BlueGlow'),
                     transitions={'succeeded': 'SWITCH_TO_BLUE7',
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SWITCH_TO_BLUE7',
                 SetMaterialColorServiceState('Box7',
                                              'body',
                                              'visual',
                                              'Gazebo/BlueGlow'),
                 transitions={'succeeded': 'SWITCH_TO_BLUE8',
                              'aborted': ERROR,
                              'preempted': PREEMPTED})

    StateMachine.add('SWITCH_TO_BLUE8',
             SetMaterialColorServiceState('Box8',
                                          'body',
                                          'visual',
                                          'Gazebo/BlueGlow'),
             transitions={'succeeded': 'WAIT',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('WAIT',
                    WaitToClockState(20),
                     {'valid': 'WAIT', 'invalid': 'END_MSG',
                      'preempted': PREEMPTED})

    StateMachine.add('END_MSG',
                     CBState(notify_user_cb('The color of the boxes is set back to WHITE to see neuronal persistent activity')),
                     {'succeeded': 'DELAY_ON_END'})

    StateMachine.add('DELAY_ON_END',
                         ClockDelayState(3),
                         transitions={'valid': 'DELAY_ON_END', 
                                      'invalid': 'SET_GRAY0', 
                                      'preempted': PREEMPTED})

    StateMachine.add('SET_GRAY0',
            SetMaterialColorServiceState('Box0',
                                          'body',
                                          'visual',
                                          'Gazebo/WhiteGlow'),
             transitions={'succeeded': 'SET_GRAY1',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('SET_GRAY1',
            SetMaterialColorServiceState('Box1',
                                          'body',
                                          'visual',
                                          'Gazebo/WhiteGlow'),
             transitions={'succeeded': 'SET_GRAY2',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('SET_GRAY2',
            SetMaterialColorServiceState('Box2',
                                          'body',
                                          'visual',
                                          'Gazebo/WhiteGlow'),
             transitions={'succeeded': 'SET_GRAY3',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('SET_GRAY3',
            SetMaterialColorServiceState('Box3',
                                          'body',
                                          'visual',
                                          'Gazebo/WhiteGlow'),
             transitions={'succeeded': 'SET_GRAY4',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('SET_GRAY4',
            SetMaterialColorServiceState('Box4',
                                          'body',
                                          'visual',
                                          'Gazebo/WhiteGlow'),
             transitions={'succeeded': 'SET_GRAY5',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('SET_GRAY5',
            SetMaterialColorServiceState('Box5',
                                          'body',
                                          'visual',
                                          'Gazebo/WhiteGlow'),
             transitions={'succeeded': 'SET_GRAY6',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('SET_GRAY6',
            SetMaterialColorServiceState('Box6',
                                          'body',
                                          'visual',
                                          'Gazebo/WhiteGlow'),
             transitions={'succeeded': 'SET_GRAY7',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('SET_GRAY7',
            SetMaterialColorServiceState('Box7',
                                          'body',
                                          'visual',
                                          'Gazebo/WhiteGlow'),
             transitions={'succeeded': 'SET_GRAY8',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('SET_GRAY8',
            SetMaterialColorServiceState('Box8',
                                          'body',
                                          'visual',
                                          'Gazebo/WhiteGlow'),
             transitions={'succeeded': 'END_MSG2',
                          'aborted': ERROR,
                          'preempted': PREEMPTED})

    StateMachine.add('END_MSG2',
                     CBState(notify_user_cb('It can be seen that the neuronal activity holds for more or less 2s. Then random firing restarts')),
                     {'succeeded': 'DELAY_ON_END2'})

    StateMachine.add('DELAY_ON_END2',
                         ClockDelayState(15),
                         transitions={'valid': 'DELAY_ON_END', 
                                      'invalid': 'WAIT2', 
                                      'preempted': PREEMPTED})

    StateMachine.add('WAIT2',
                    WaitToClockState(10),
                     {'valid': 'WAIT2', 'invalid': FINISHED,
                      'preempted': PREEMPTED})

