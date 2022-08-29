## Demonstration Generation

If you want to use our demonstration generation pipeline for your research, you can check this README file.
Note that it does not contain information the object and hand pose estimation algorithm. 
It assumes you already get the pose results using any pose estimation algorithm.

## Retargeting

An [example retargeting code](../examples/retarget_human_hand.py) is already introduced in the main [README.md](../README.md)
However, it does not consider the extrinsics of camera during retargeting, so the retargeting results are represented in the camera space, not world space.


## World-Camera Transformation

To align with the [environment](env.md), where observation(state) are represented in the world space, we need to transform both the human hand pose and object pose using camera extrinsics, before retargeting.
In DexMV, we measure the pose of table with respect to camera. Which is fixed during the whole data collection process.

## Trajectory Interpolation/Time Parameterization

To use inverse dynamics functions, we need to first compute the velocity and acceleration of robot joint. In this step, time step is aligned between real world measurement and simulation timestep.
For more details, please check the code in [this directory](../hand_imitation/kinematics/demonstration)

## Inverse Dynamics

The inverse dynamics is computed via the MuJoCo API. The input is joint position, velocity, and acceleration. The output is joint torque.
For more details, please check [this file](../hand_imitation/kinematics/demonstration/base.py)

## Hindsight Target
Relocate is a goal-oriented task. For demonstration, we set the object pose in last frame as the target pose(in face we only consider position)
For more details, please check [this file](../hand_imitation/kinematics/demonstration/relocation_demo.py)






