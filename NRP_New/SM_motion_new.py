##
#!/usr/bin/env python
"""

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
    SetMaterialColorServiceState, ClockDelayState, SpawnSphere, SpawnBox, DestroyModel, SetModelPose, \
    TranslateModelState
from hbp_nrp_excontrol.logs import clientLogger
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, ModelStates

################################################################
#################### FUNCTION DEFINITION #######################
################################################################

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

        msg.name = name
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


def moveAlongPath(models, pointFrom, pointTo, stepSize=0.1):
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
        move_object_cb(models, newPos)(userdata)

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

initialSphere0Pos = Point(-0.5, 0, 0.3)
targetSphere0Pos = Point(initialSphere0Pos.x, initialSphere0Pos.y + 2, initialSphere0Pos.z)

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
                                color='Gazebo/Black'),
                     transitions={'succeeded': 'SPAWN_OBJECT1', 
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})
    
    StateMachine.add('SPAWN_OBJECT1',
                     SpawnBox(model_name="Box1", size = Vector3(1,1,1),
                                 position=Point(-1,1,0.5), gravity_factor=0,
                                color='Gazebo/Black'),
                     transitions={'succeeded': 'SPAWN_OBJECT2', 
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})
    
    StateMachine.add('SPAWN_OBJECT2',
                 SpawnBox(model_name="Box2", size = Vector3(1,1,1),
                             position=Point(-1,-1,0.5), gravity_factor=0,
                            color='Gazebo/Black'),
                 transitions={'succeeded': 'SPAWN_OBJECT3', 
                              'aborted': ERROR,
                              'preempted': PREEMPTED})
    
    StateMachine.add('SPAWN_OBJECT3',
                     SpawnBox(model_name="Box3", size = Vector3(1,1,1),
                                 position=Point(-1,0,1.5), gravity_factor=0,
                                color='Gazebo/Black'),
                     transitions={'succeeded': 'SPAWN_OBJECT4', 
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})
    
    StateMachine.add('SPAWN_OBJECT4',
                     SpawnBox(model_name="Box4", size = Vector3(1,1,1),
                                 position=Point(-1,1,1.5), gravity_factor=0,
                                color='Gazebo/Black'),
                     transitions={'succeeded': 'SPAWN_OBJECT5', 
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})
    
    StateMachine.add('SPAWN_OBJECT5',
                     SpawnBox(model_name="Box5", size = Vector3(1,1,1),
                                 position=Point(-1,-1,1.5), gravity_factor=0,
                                color='Gazebo/Black'),
                     transitions={'succeeded': 'SPAWN_OBJECT6', 
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SPAWN_OBJECT6',
                     SpawnBox(model_name="Box6",size = Vector3(1,1,1),
                                 position=Point(-1,0,2.5), gravity_factor=0,
                                color='Gazebo/Black'),
                     transitions={'succeeded': 'SPAWN_OBJECT7', 
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})
     
    StateMachine.add('SPAWN_OBJECT7',
                     SpawnBox(model_name="Box7",size = Vector3(1,1,1),
                                 position=Point(-1,1,2.5), gravity_factor=0,
                                color='Gazebo/Black'),
                     transitions={'succeeded': 'SPAWN_OBJECT8', 
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SPAWN_OBJECT8',
                     SpawnBox(model_name="Box8",size = Vector3(1,1,1),
                                 position=Point(-1,-1,2.5), gravity_factor=0,
                                color='Gazebo/Black'),
                     transitions={'succeeded': 'SPAWN_OBJECT9',     
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SPAWN_OBJECT9',
                         SpawnBox(model_name="Box9",size = Vector3(1,1,1),
                                     position=Point(-1,2,0.5), gravity_factor=0,
                                    color='Gazebo/Black'),
                         transitions={'succeeded': 'SPAWN_OBJECT10', 
                                      'aborted': ERROR,
                                      'preempted': PREEMPTED})
     
    StateMachine.add('SPAWN_OBJECT10',
                     SpawnBox(model_name="Box10",size = Vector3(1,1,1),
                                 position=Point(-1,2,1.5), gravity_factor=0,
                                color='Gazebo/Black'),
                     transitions={'succeeded': 'SPAWN_OBJECT11', 
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SPAWN_OBJECT11',
                     SpawnBox(model_name="Box11",size = Vector3(1,1,1),
                                 position=Point(-1,2,2.5), gravity_factor=0,
                                color='Gazebo/Black'),
                     transitions={'succeeded': 'SPAWN_OBJECT12',     
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SPAWN_OBJECT12',
                         SpawnBox(model_name="Box12",size = Vector3(1,1,1),
                                     position=Point(-1,-2,0.5), gravity_factor=0,
                                    color='Gazebo/Black'),
                         transitions={'succeeded': 'SPAWN_OBJECT13', 
                                      'aborted': ERROR,
                                      'preempted': PREEMPTED})
     
    StateMachine.add('SPAWN_OBJECT13',
                     SpawnBox(model_name="Box13",size = Vector3(1,1,1),
                                 position=Point(-1,-2,1.5), gravity_factor=0,
                                color='Gazebo/Black'),
                     transitions={'succeeded': 'SPAWN_OBJECT14', 
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SPAWN_OBJECT14',
                     SpawnBox(model_name="Box14",size = Vector3(1,1,1),
                                 position=Point(-1,-2,2.5), gravity_factor=0,
                                color='Gazebo/Black'),
                     transitions={'succeeded': 'SPAWN_SPHERE0',     
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SPAWN_SPHERE0',
                    SpawnSphere(model_name='Sphere0', radius=0.2,
                                position = Point(-0.5, 0, 0.3), gravity_factor=0,
                                color='Gazebo/Grey'),
                    transitions={'succeeded': 'SPAWN_SPHERE1',     
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SPAWN_SPHERE1',
                    SpawnSphere(model_name='Sphere1', radius=0.2,
                                position = Point(-0.5,0,0.7), gravity_factor=0,
                                color='Gazebo/Grey'),
                    transitions={'succeeded': 'SPAWN_SPHERE2',     
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SPAWN_SPHERE2',
                    SpawnSphere(model_name='Sphere2', radius=0.2,
                                position = Point(-0.5,0,1.1), gravity_factor=0,
                                color='Gazebo/Grey'),
                    transitions={'succeeded': 'SPAWN_SPHERE3',     
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SPAWN_SPHERE3',
                    SpawnSphere(model_name='Sphere3', radius=0.2,
                                position = Point(-0.5,0,1.5), gravity_factor=0,
                                color='Gazebo/Grey'),
                    transitions={'succeeded': 'SPAWN_SPHERE4',     
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SPAWN_SPHERE4',
                    SpawnSphere(model_name='Sphere4', radius=0.2,
                                position = Point(-0.5,0,1.9), gravity_factor=0,
                                color='Gazebo/Grey'),
                    transitions={'succeeded': 'SPAWN_SPHERE5',     
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SPAWN_SPHERE5',
                    SpawnSphere(model_name='Sphere5', radius=0.2,
                                position = Point(-0.5,0,2.3), gravity_factor=0,
                                color='Gazebo/Grey'),
                    transitions={'succeeded': 'SPAWN_SPHERE6',     
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('SPAWN_SPHERE6',
                    SpawnSphere(model_name='Sphere6', radius=0.2,
                                position = Point(-0.5,0,2.7), gravity_factor=0,
                                color='Gazebo/Grey'),
                    transitions={'succeeded': 'START_MSG',     
                                  'aborted': ERROR,
                                  'preempted': PREEMPTED})

    StateMachine.add('START_MSG',
                     CBState(notify_user_cb('The spheres start to move and the robot will produce a motor response')),
                     {'succeeded': 'DELAY_ON_MOTION0'})

    StateMachine.add('DELAY_ON_MOTION0',
                         ClockDelayState(1),
                         transitions={'valid': 'DELAY_ON_MOTION0', 
                                      'invalid': 'MOVE_SPHERE0',
                                      'preempted': PREEMPTED})

    StateMachine.add('MOVE_SPHERE0',
                         CBState(
                            moveAlongPath(['Sphere0','Sphere1','Sphere2','Sphere3','Sphere4','Sphere5','Sphere6'], 
                                initialSphere0Pos, targetSphere0Pos, 0.005)),
                         transitions={'succeeded': 'END_MSG',
                                      'ongoing': 'MOVE_SPHERE0'})

    StateMachine.add('END_MSG',
                     CBState(notify_user_cb('The experiment is finished')),
                     {'succeeded': 'WAIT'})

    StateMachine.add('WAIT',
                    WaitToClockState(10),
                     {'valid': 'WAIT', 'invalid': FINISHED,
                      'preempted': PREEMPTED})