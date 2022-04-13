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




#TODO

Need to take reference trajectories data for the above map and train with the SAC.



