""" 

# -*- coding: utf-8 -*-



Created on Mon April 11 2022

@author: Rahul Karanam
@brief Train a SAC agent for the High Speed Driving on CARLA simulator. ( I'll be training on the map - 1 which
can be found in the reference folder )


"""

# Importing the libraries
import sys
from SoftAC import *
import time
import numpy as np
import random
import pygame
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from agents.navigation.basic_agent import BasicAgent


if __name__ == "__main__":
    
    # To check pygame is working or not
    print("Working:[1]")
    pygame.init()
    print("Working:[2]")
    pygame.font.init()
    print("Working:[3]")
    env = environment(traj_num = 1)
