#!/usr/bin/python3
import numpy as np
from utils import Model

- controller class
    - init 
    - load_model
    - .vehicle_state (class)
    - .act_history
    - .obs_history
    - .state_size
    - .act_size
    - .last_act
    
# inputs current raw data (human-readable, non-normalized), outputs actual motor speed
# TODO: run() func 
# TODO: pass Vehicle_State class instance 
# TODO: return motor speed 
# TODO: define motor speed model in comments 
# TODO: keep running memory of obs history and act history
class AdapLowLevelControl:
    def __init__(self):

        # time
        self.t = 0

        # base model