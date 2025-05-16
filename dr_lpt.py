import numpy as np
import torch

from torchrl.envs.utils import check_env_specs, step_mdp

from jsp_env import JSPEnv

def LPT_solver(instance=None):
    env = JSPEnv(instance)
    state = env.reset()
    
    done = False
    while not done:
        real_obs = state["real_obs"]
        action_mask = state["action_mask"][:-1]
        LPT_action = 0
        max_operation_time = -1
        for job in range(env.jobs):
            if action_mask[job]:
                needed_machine = env.jobs_todo_time_step[job]
                operation_time = env.instance_matrix[job][needed_machine][1]
                if operation_time > max_operation_time:
                    max_operation_time = operation_time
                    LPT_action = job
        LPT_action = torch.tensor(LPT_action)
        state_with_action = state.clone()
        state_with_action["action"] = LPT_action
        stepped_data = env.step(state_with_action)
        state = step_mdp(stepped_data)
        done = state["done"]
    env.reset()
    return env.last_time_step