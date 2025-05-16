import numpy as np

from torchrl.envs.utils import check_env_specs, step_mdp

from jsp_env import JSPEnv

def MWTR_solver(instance=None):
    env = JSPEnv(instance)
    state = env.reset()
    
    done = False
    while not done:
        real_obs = state["real_obs"]
        action_mask = state["action_mask"][:-1]
        reshaped = np.reshape(real_obs, (env.jobs, 7))
        remaining_time = (reshaped[:, 3] * env.max_time_job) / env.jobs_time
        illegal_actions = np.invert(action_mask)
        mask = illegal_actions * 1e8
        remaining_time += mask
        MWTR_action = np.argmin(remaining_time)
        state_with_action = state.clone()
        state_with_action["action"] = MWTR_action
        stepped_data = env.step(state_with_action)
        state = step_mdp(stepped_data)
        done = state["done"]
    env.reset()
    return env.last_time_step    