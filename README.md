# Autonomous-Drifting-using-deep-Reinforcement-Learning
A Soft Actor Policy based model free off-policy network to control the steering and throttle of car while drifting at high speeds.


- Our model will be trained using the Soft-Actor Critic
(SAC), which optimizes the error loss (anticipated return - prediction) and maximizes entropy using an off-policy
learning strategy to perform better in continuous domains.


- Control policy is the actor in SAC, while value and
Q-network will function as critics.


- The basic goal of the actor is to maximize reward while
minimizing entropy (measure of randomness in the policy - more exploration)

## Map for training
![](https://github.com/karanamrahul/Autonomous-Drifting-using-deep-Reinforcement-Learning/blob/main/results/track2.png)

## Basic Demo of Simualtor
![](https://github.com/karanamrahul/Autonomous-Drifting-using-deep-Reinforcement-Learning/blob/main/results/train3.png)



## Environment

- Ubuntu 20.04
- Conda : Package and environment manager
- Python 3.8
- Pytorch
- Pygame

## Installation steps for CARLA simulator

We are using CARLA [0.9.5](https://carla.readthedocs.io/en/0.9.5/getting_started/) as our version for our simulation.

Please download the the simulator from this [drive](https://drive.google.com/file/d/1CefYTLF48YKU5sPkQXsCScsG3fRiY0Gv/view?usp=sharing)

Extract the folder in your Downloads directory.

If you have a dual GPU setup , please enter the following command to enable your secondary graphics card as the primary one.
```
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"
```

```
Add Path to your bash file

export PYTHONPATH=$PYTHONPATH:~/Downloads/CARLA_DRIFT_0.9.5/PythonAPI/carla/dist/carla-0.9.5-py3.5-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:~/Downloads/CARLA_DRIFT_0.9.5/PythonAPI/carla/

```


To open and run the simulator please enter the below commands
open a new terminal
```
cd Downloads/CARLA_DRIFT_0.9.5
./CarlaUE4.sh /Game/Carla/ExportedMaps/simple
```

This will show up the map.If you want to spawn vehicles and manually control the vehicle in the above 
map please enter the below commands.

```
Open a New terminal
cd Downloads/CARLA_DRIFT_0.9.5/PythonAPI/examples
./spawn_npc.py
```

This will spawn vehicels in the map.

To control a vehicle in the environment, enter the below commands.
```
Open a New terminal
cd Downloads/CARLA_DRIFT_0.9.5/PythonAPI/examples
./manual_control.py
```



#TODO

Need to take reference trajectories data for the above map and train with the SAC.



