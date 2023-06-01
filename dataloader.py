import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import logging

LOG = logging.getLogger('base')

# constant 
state_obs_size = 17
act_size = 4
history_len = 400
latent_size = 8

state_vars = ['ori1', 'ori2', 'ori3', 'ori4', 'ori5', 'ori6', 'ori7', 'ori8', 'ori9', 'wx', 'wy', 'wz', 'prop_acc', 'cmd_wx', 'cmd_wy', 'cmd_wz', 'cmd_prop_acc']
action_vars = ['act1', 'act2', 'act3', 'act4']
env_params = ['mass', 'arm_length', 'Jxx', 'Jyy', 'Jzz', 'ext_torque_x', 'ext_torque_y', 'ext_torque_z', 'kappa', 'thrustRatioSpdSq', 'bodyDrag1Coeffx', 'bodyDrag1Coeffy', 'bodyDrag1Coeffz', 'motorSpdMax', 'B_allocate_1', 'B_allocate_2', 'B_allocate_3', 'B_allocate_4', 'B_allocate_5', 'B_allocate_6', 'B_allocate_7', 'B_allocate_8', 'B_allocate_9', 'B_allocate_10', 'B_allocate_11', 'B_allocate_12', 'B_allocate_13', 'B_allocate_14', 'B_allocate_15', 'B_allocate_16', 'corr_fac1', 'corr_fac2', 'corr_fac3', 'corr_fac4']

# normalization 
rms_dir = "data/model_RMS.npz"
rms_data = np.load(rms_dir)
obs_mean = np.mean(rms_data["mean"], axis=0)
obs_var = np.mean(rms_data["var"], axis=0)

def normalize_obs(obs_n_norm):
        
        obs_n_norm = obs_n_norm.reshape([1, -1])

        obs_state_history_n_normalized = obs_n_norm[:, :-history_len*act_size]

        obs_state_mean = np.tile(obs_mean[:state_obs_size], [1,history_len])
        obs_state_var = np.tile(obs_var[:state_obs_size], [1, history_len])
        
        # Only normalize state history, Not act history  
        obs_state_history_normalized = (
            obs_state_history_n_normalized - obs_state_mean) / np.sqrt(obs_state_var + 1e-8)

        obs_n_norm[:,:-history_len*act_size] = obs_state_history_normalized
       

        return obs_n_norm


class TrajDataset(Dataset):
    def __init__(self, path='./data/5000Traj_collection_Policy248.csv', std_threshold=0):        
        LOG.info(f"Loading data from {path}...")
        df = df = pd.read_csv(path)
        num_episodes = len(df['episode_id'].unique())
        episode_length = len(df) // num_episodes
        LOG.info(f"Loaded {len(df)} rows, {num_episodes} episodes, each of length {episode_length}")
        env_std = np.std(df[env_params].values, axis=0)
        cols = np.where(env_std >= std_threshold)[0]
        LOG.info(f"Using threshold {std_threshold} to select, chose a total of {len(cols)} env params out of {len(env_params)}")
        env = df[env_params].values[:, cols]
        self.env_max_scale = (np.max(np.abs(env), axis=0) + 1e-6)
        env = env / self.env_max_scale
        state = df[state_vars].values
        act = df[action_vars].values
        LOG.info(f"state shape: {state.shape}, act shape: {act.shape}, env shape: {env.shape}")
        
        self.state = state.astype(np.float32)
        self.env_param_dims = env.shape[1]
        self.act = act.astype(np.float32)
        self.env = env.astype(np.float32)
        self.length = len(state)
        self.num_episodes = num_episodes
        self.episode_length = episode_length

        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        state = np.zeros((history_len, self.state.shape[1]), dtype=np.float32)
        act = np.zeros((history_len, self.act.shape[1]), dtype=np.float32)

        length_since_episode_start = (idx % self.episode_length) + 1 # How many rows since the start of the episode
        copy_length = min(length_since_episode_start, history_len) # How many rows to copy
        state[-copy_length:] = self.state[idx+1-copy_length:idx+1]
        act[-copy_length:] = self.act[idx+1-copy_length:idx+1]
        env = self.env[idx]
        
        obs = np.concatenate([state.flatten(), act.flatten()], axis=0)
        obs_norm = normalize_obs(obs).flatten()
        env = env.reshape(-1).flatten()
        return torch.from_numpy(env).float(), torch.from_numpy(obs_norm).float()