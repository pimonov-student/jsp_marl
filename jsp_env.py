from pathlib import Path
import datetime
import bisect
import random

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.data import (
    Bounded,
    Composite,
    Unbounded,
    Categorical)
from torchrl.envs import (
    EnvBase,
)

def make_composite_from_td(td):
    composite = Composite(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else Unbounded(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite

# def gen_info():
#     jobs = torch.randint(3, 40, (1,))
#     machines = torch.randint(3, 40, (1,))

class JSPEnv(EnvBase):
    _batch_locked = False
    
    def __init__(self, td_params=None, seed=None, device="cpu"):
        self.jobs = 0
        self.machines = 0
        self.instance_matrix = None
        
        self.last_time_step = float("inf")
        self.current_time_step = float("inf")
        self.next_time_step = list()
        
        self.start_timestamp = datetime.datetime.now().timestamp()
        
        self.max_time_operation = 0
        self.max_time_job = 0
        self.operations_total_time = 0
        
        self.jobs_next = list()
        self.jobs_time = None
        self.jobs_needed_machine = None
        self.jobs_time_until_complete_current_operation = None
        self.jobs_todo_time_step = None
        self.jobs_total_time_completed_operations = None
        self.jobs_total_idle_time = None
        self.jobs_idle_time_last_operation = None
        
        self.actions_legal = None
        self.actions_legal_number = 0
        self.actions_illegal = None
        self.actions_illegal_no_operations = None
        
        self.machines_legal = None
        self.machines_legal_number = 0
        self.machines_time_until_free = None
        
        self.state = None
        
        self.solution = None
        self.colors = None
        
        if td_params is None:
            td_params = {
                "instance_path": Path(__file__).parent.absolute() / "instances" / "ta01"
            }
        self._set_info("read", td_params["instance_path"])
        
        super().__init__(device=device, batch_size=[])
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self._set_seed(seed)
    
    def _set_info(self, mode, instance_path=None):
        if mode == "gen":
            None
        else:
            self._read_info(instance_path)
    
    def _read_info(self, instance_path):
        with open(instance_path, "r") as instance_file:
            for line_cnt, line in enumerate(instance_file, start=1):
                split_data = list(map(int, line.split()))
                # first line with number of JOBS and MACHINES
                if line_cnt == 1:
                    self.jobs, self.machines = split_data
                    self.instance_matrix = np.zeros((self.jobs, self.machines), dtype=(int, 2))
                    self.jobs_time = np.zeros(self.jobs, dtype=int)
                    self.colors = [
                        tuple([random.random() for _ in range(3)]) for _ in range(self.machines)
                    ]
                # other lines with per job data
                else:
                    job_number = line_cnt - 2
                    for i in range(0, len(split_data), 2):
                        machine, time = split_data[i], split_data[i + 1]
                        self.instance_matrix[job_number][i // 2] = (machine, time)
                        self.max_time_operation = max(self.max_time_operation, time)
                        self.jobs_time[job_number] += time
                        self.operations_total_time += time
        self.max_time_job = max(self.jobs_time)
    
    def _make_spec(self):
        # observation
        self.observation_spec = Composite(
            action_mask = Bounded(
                low=False,
                high=True,
                shape=(self.jobs + 1,),
                dtype=torch.bool,
                device="cpu",
            ),
            real_obs = Unbounded(
                shape=(self.jobs, 7),
                dtype=torch.float32,
                device="cpu",
            ),
            shape=(),
        )
        # state
        self.state_spec = self.observation_spec.clone()
        # action
        self.action_spec = Categorical(
            n=self.jobs + 1,
            shape=(1,),
            dtype=torch.int64,
            device="cpu",
        )
        # reward
        self.reward_spec = Unbounded(shape=(1,))
    
    def _reset(self, tensordict):
        self.current_time_step = 0
        self.next_time_step = list()
        
        self.jobs_next = list()
        self.jobs_needed_machine = np.zeros(self.jobs, dtype=np.int32)
        self.jobs_time_until_complete_current_operation = np.zeros(self.jobs, dtype=np.int32)
        self.jobs_todo_time_step = np.zeros(self.jobs, dtype=np.int32)
        self.jobs_total_time_completed_operations = np.zeros(self.jobs, dtype=np.int32)
        self.jobs_total_idle_time = np.zeros(self.jobs, dtype=np.int32)
        self.jobs_idle_time_last_operation = np.zeros(self.jobs, dtype=np.int32)
        
        self.actions_legal = np.ones(self.jobs + 1, dtype=bool)
        self.actions_legal[self.jobs] = False
        self.actions_legal_number = self.jobs
        self.actions_illegal = np.zeros((self.machines, self.jobs), dtype=bool)
        self.actions_illegal_no_operations = np.zeros(self.jobs, dtype=bool)
        
        self.machines_legal = np.zeros(self.machines, dtype=bool)
        self.machines_legal_number = 0
        self.machines_time_until_free = np.zeros(self.machines, dtype=np.int32)
        
        self.state = np.zeros((self.jobs, 7), dtype=np.float32)
        
        self.solution = np.full((self.jobs, self.machines,), -1, dtype=int)
        
        for job in range(self.jobs):
            needed_machine = self.instance_matrix[job][0][0]
            self.jobs_needed_machine[job] = needed_machine
            if not self.machines_legal[needed_machine]:
                self.machines_legal[needed_machine] = True
                self.machines_legal_number += 1
        
        return self._get_state()
    
    def _step(self, tensordict):
        reward = 0.0
        action = tensordict["action"].item()
        # idle action case
        if action == self.jobs:
            self.actions_legal_number = 0
            self.machines_legal_number = 0
            for job in range(self.jobs):
                if self.actions_legal[job]:
                    self.actions_legal[job] = False
                    needed_machine = self.jobs_needed_machine[job]
                    self.machines_legal[needed_machine] = False
                    self.actions_illegal[needed_machine][job] = True
                    self.actions_illegal_no_operations[job] = True
            while self.machines_legal_number == 0:
                reward -= self._time_step()
            scaled_reward = self._scale_reward(reward)
            
            state = self._get_state()
            out = TensorDict(
                {
                    "action_mask": state["action_mask"],
                    "real_obs": state["real_obs"],
                    "reward": scaled_reward,
                    "done": self._is_done(),
                },
                tensordict.shape,
            )
            return out
        # not idle action case
        else:
            # get current operation id of job
            current_time_step_job = self.jobs_todo_time_step[action]
            # get current operation needed machine
            machine_needed = self.jobs_needed_machine[action]
            # get time, needed to complete current operation
            time_needed = self.instance_matrix[action][current_time_step_job][1]
            reward += time_needed
            self.machines_time_until_free[machine_needed] = time_needed
            self.jobs_time_until_complete_current_operation[action] = time_needed
            self.state[action][1] = time_needed / self.max_time_operation
            to_add_time_step = self.current_time_step + time_needed
            # if time step, when current operation will be finished
            # is not in list next_time_step, we add it to the list 
            if to_add_time_step not in self.next_time_step:
                index = bisect.bisect_left(self.next_time_step, to_add_time_step)
                self.next_time_step.insert(index, to_add_time_step)
                self.jobs_next.insert(index, action)
            # set current operation start time to solution
            self.solution[action][current_time_step_job] = self.current_time_step
            # for each job, which current operation needs the occupied machine
            # we set those jobs choice as illegal
            for job in range(self.jobs):
                if (self.jobs_needed_machine[job] == machine_needed
                    and self.actions_legal[job]):
                    self.actions_legal[job] = False
                    self.actions_legal_number -= 1
            self.machines_legal_number -= 1
            # occupied machine is unavailable for now
            self.machines_legal[machine_needed] = False
            for job in range(self.jobs):
                if self.actions_illegal[machine_needed][job]:
                    self.actions_illegal_no_operations[job] = False
                    self.actions_illegal[machine_needed][job] = False
            # if all machines are busy, we skip time in _time_step()
            # _time_step returns time amount of "time holes", in which jobs are not completing
            while self.machines_legal_number == 0 and len(self.next_time_step) > 0:
                reward -= self._time_step()
            scaled_reward = self._scale_reward(reward)
            state = self._get_state()
            out = TensorDict(
                {
                    "action_mask": state["action_mask"],
                    "real_obs": state["real_obs"],
                    "reward": scaled_reward,
                    "done": self._is_done(),
                },
                tensordict.shape,
            )
            return out
    
    def _time_step(self):
        hole_planning = 0
        # shortest time, in which any job's current operation will be completed
        next_time_step_to_pick = self.next_time_step.pop(0)
        self.jobs_next.pop(0)
        # time difference between current time and time, when operation will be completed
        difference = next_time_step_to_pick - self.current_time_step
        # update current time step
        self.current_time_step = next_time_step_to_pick
        # JOB WORK
        for job in range(self.jobs):
            # time until current job's operation will be completed
            left_time = self.jobs_time_until_complete_current_operation[job]
            # if job's current operation is in progress
            if left_time > 0:
                # time to spend on completing the operation
                job_performed_op = min(difference, left_time)
                self.jobs_time_until_complete_current_operation[job] = max(
                    0, self.jobs_time_until_complete_current_operation[job] - difference
                )
                self.state[job][1] = (
                    self.jobs_time_until_complete_current_operation[job] / self.max_time_operation
                )
                self.jobs_total_time_completed_operations[job] += job_performed_op
                self.state[job][3] = (
                    self.jobs_total_time_completed_operations[job] / self.max_time_job
                )
                # if operation completed during time_step
                if self.jobs_time_until_complete_current_operation[job] == 0:
                    # check "hole time" 
                    self.jobs_total_idle_time[job] += difference - left_time
                    self.state[job][6] = self.jobs_total_idle_time[job] /  self.operations_total_time
                    self.jobs_idle_time_last_operation[job] = difference - left_time
                    self.state[job][5] = self.jobs_idle_time_last_operation[job] / self.operations_total_time
                    # go to next operation id
                    self.jobs_todo_time_step[job] += 1
                    self.state[job][2] = self.jobs_todo_time_step[job] / self.machines
                    # if job is not completed
                    if self.jobs_todo_time_step[job] < self.machines:
                        # get new needed machine
                        self.jobs_needed_machine[job] = self.instance_matrix[job][self.jobs_todo_time_step[job]][0]
                        self.state[job][4] = (
                            max(0, self.machines_time_until_free[self.jobs_needed_machine[job]] - difference) / self.max_time_operation
                        )
                    # if job is completed
                    else:
                        # job need no machine
                        self.jobs_needed_machine[job] = -1
                        self.state[job][4] = 1.0
                        # make action choice of this job illegal
                        if self.actions_legal[job]:
                            self.actions_legal[job] = False
                            self.actions_legal_number -= 1
            # if job's current operation is not in progress
            elif self.jobs_todo_time_step[job] < self.machines:
                self.jobs_total_idle_time[job] += difference
                self.jobs_idle_time_last_operation[job] += difference
                self.state[job][5] = self.jobs_idle_time_last_operation[job] / self.operations_total_time
                self.state[job][6] = self.jobs_total_idle_time[job] / self.operations_total_time
        # MACHINE WORK
        for machine in range(self.machines):
            # if machine will be free in time step
            if self.machines_time_until_free[machine] < difference:
                # calculate time, for which machine won't be doing anything
                empty = difference - self.machines_time_until_free[machine]
                hole_planning += empty
            self.machines_time_until_free[machine] = max(0, self.machines_time_until_free[machine] - difference)
            # if machine is free
            if self.machines_time_until_free[machine] == 0:
                # for each job make current machine free to use
                for job in range(self.jobs):
                    if (self.jobs_needed_machine[job] == machine
                        and not self.actions_legal[job]
                        and not self.actions_illegal[machine][job]):
                        self.actions_legal[job] = True
                        self.actions_legal_number += 1
                        # make machine legal to choose
                        if not self.machines_legal[machine]:
                            self.machines_legal[machine] = True
                            self.machines_legal_number += 1
        return hole_planning      
        
    def _scale_reward(self, reward):
        return np.float32(reward / self.max_time_operation)
    
    def _is_done(self):
        if self.actions_legal_number == 0:
            self.last_time_step = self.current_time_step
            return True
        return False
    
    def _get_state(self):
        self.state[:, 0] = self.actions_legal[:-1]
        out = TensorDict(
            {
                "action_mask": self.actions_legal,
                "real_obs": self.state,
            },
            [],
        )
        return out
     
    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng
    
    def render(self):
        df = []
        for job in range(self.jobs):
            i = 0
            while i < self.machines and self.solution[job][i] != -1:
                op = dict()
                op["Task"] = "Job {}".format(job)
                start = self.start_timestamp + self.solution[job][i]
                finish = start + self.instance_matrix[job][i][1]
                op["Start"] = datetime.datetime.fromtimestamp(start)
                op["Finish"] = datetime.datetime.fromtimestamp(finish)
                op["Color"] = "Machine {}".format(
                    self.instance_matrix[job][i][0]
                )
                df.append(op)
                i += 1
        return pd.DataFrame(df)

# from torchrl.envs import (
#     EnvBase,
#     TransformedEnv,
#     ActionMask,
# )
# from torchrl.envs.utils import step_mdp

# base_env = JSPEnv()
# env = TransformedEnv(base_env, ActionMask())

# state = env.reset()
# done = False
# while not done:
#     step = env.rand_step(state)
#     print(state["real_obs"])
#     state = step_mdp(step)
#     done = state["done"]
#     print(done)
