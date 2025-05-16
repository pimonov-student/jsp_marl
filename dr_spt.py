import numpy as np
import torch

from torchrl.envs.utils import check_env_specs, step_mdp

from jsp_env import JSPEnv

def SPT_solver(instance=None):
    env = JSPEnv(instance)
    state = env.reset()
    
    done = False
    while not done:
        real_obs = state["real_obs"]
        action_mask = state["action_mask"][:-1]
        SPT_action = 0
        min_operation_time = 1000
        for job in range(env.jobs):
            if action_mask[job]:
                needed_machine = env.jobs_todo_time_step[job]
                operation_time = env.instance_matrix[job][needed_machine][1]
                if operation_time < min_operation_time:
                    min_operation_time = operation_time
                    SPT_action = job
        SPT_action = torch.tensor(SPT_action)
        state_with_action = state.clone()
        state_with_action["action"] = SPT_action
        stepped_data = env.step(state_with_action)
        state = step_mdp(stepped_data)
        done = state["done"]
    env.reset()
    return env.last_time_step