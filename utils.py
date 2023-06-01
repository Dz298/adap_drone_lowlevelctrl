#!/usr/bin/python3

import numpy as np
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
import onnx
from onnx import numpy_helper
from enum import Enum

class ModelType(Enum):
    BASE_MODEL = 1
    ADAP_MODULE = 2
   
    
# do inference, activate session, get inputs name
# normalize obs, unnormalize inputs

# TODO: ONNX model conversion, check validity 
# TODO: where to put const size??? 
# TODO: combine base model and adaptive module together!!
# TODO: Unit Test

class Model:
    def __init__(self, model_path='./best_so_far/base_model.onnx', model_rms_path='./best_so_far/base_model.npz', model_type=ModelType.BASE_MODEL):
        self.model_path = model_path
        self.model_rms_path = model_rms_path
        self.model_type = model_type
        
        # TODO: base model + adap module def 
        
        rms_data = np.load(self.model_rms_path)
        self.obs_mean = np.mean(rms_data["mean"], axis=0)
        self.obs_var = np.mean(rms_data["var"], axis=0)
        
        self.act_mean =  np.array([1.0 / 2, 1.0 / 2,
                        1.0 / 2, 1.0 / 2])[np.newaxis, :]
        self.act_std = np.array([1.0 / 2, 1.0 / 2,
                        1.0 / 2, 1.0 / 2])[np.newaxis, :] 
        
        self.session = None
        self.obs_name = None

    def activate(self):
        self.session = onnxruntime.InferenceSession(self.model_path, None)
        self.obs_name = self.session.get_inputs()[0].name  

    def normalize_obs(self, obs):
        if self.model_type is ModelType.BASE_MODEL:
            return (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
        else:
            # Normalize for Adaptation module observations 
            obs_n_norm = obs.reshape([1, -1])

            obs_current_n_normalized = obs_n_norm[:,
                                          :-history_len*(act_size+state_obs_size)]
            obs_current_normalized = (obs_current_n_normalized - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
            obs_n_norm[:, :-history_len *
               (act_size+state_obs_size)] = obs_current_normalized

            obs_state_history_n_normalized = obs_n_norm[:, -history_len*(
                act_size+state_obs_size):-history_len*act_size]

            obs_state_mean = np.tile(obs_mean[:state_obs_size], [1, history_len])
            obs_state_var = np.tile(obs_var[:state_obs_size], [1, history_len])

            obs_state_history_normalized = (
                obs_state_history_n_normalized - obs_state_mean) / np.sqrt(obs_state_var + 1e-8)

            obs_n_norm[:, -history_len*(act_size+state_obs_size):-history_len*act_size] = obs_state_history_normalized

            obs_norm = obs_n_norm
            
            return obs_norm
        
    def run(self,obs):
        self.normalize_obs(obs)
        action = self.session.run(None, {self.obs_name: obs})
        norm_action = (action * act_std + act_mean)[0, :]
        return norm_action


class QuadState:
    def __init__(self):

        # time
        self.t = 0

        # position
        self.pos = np.array([0, 0, 0], dtype=np.float32)

        # quaternion [w,x,y,z]
        self.att = np.array([0, 0, 0, 0], dtype=np.float32)

        # velocity
        self.vel = np.array([0, 0, 0], dtype=np.float32)

        # angular velocity i.e. body rates
        self.omega = np.array([0, 0, 0], dtype=np.float32)

        # proper acceleration i.e. acceleration - G_vec
        self.proper_acc = np.array([0.0, 0.0, 0.0])

        # commanded mass-normalized thrust, from a high-level controller
        self.cmd_collective_thrust = np.array([0.0])

        # commanded angular velocity i.e. body rates, from a high-level controller
        self.cmd_bodyrates = np.array([0.0, 0.0, 0.0])

    def __repr__(self):
        repr_str = "QuadState:\n" \
                   + " t:     [%.2f]\n" % self.t \
                   + " pos:   [%.2f, %.2f, %.2f]\n" % (self.pos[0], self.pos[1], self.pos[2]) \
                   + " att:   [%.2f, %.2f, %.2f, %.2f]\n" % (self.att[0], self.att[1], self.att[2], self.att[3]) \
                   + " vel:   [%.2f, %.2f, %.2f]\n" % (self.vel[0], self.vel[1], self.vel[2]) \
                   + " omega: [%.2f, %.2f, %.2f]\n" % (self.omega[0], self.omega[1], self.omega[2])\
                   + " proper_acc: [%.2f, %.2f, %.2f]\n" % (self.proper_acc[0], self.proper_acc[1], self.proper_acc[2])\
                   + " cmd_collective_thrust: [%.2f]\n" % (self.cmd_collective_thrust[0])\
                   + " cmd_bodyrates: [%.2f, %.2f, %.2f]\n" % (
                       self.cmd_bodyrates[0], self.cmd_bodyrates[1], self.cmd_bodyrates[2])
        return repr_str
