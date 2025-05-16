from pathlib import Path
import datetime
import bisect
import random

import numpy as np
import torch
from tensordict import TensorDict

from torchrl.data import (
    Bounded,
    Composite,
    Unbounded,
    Categorical)
from torchrl.envs import (
    EnvBase,
)

class MAJSPEnv(EnvBase):
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
        self.actions_legal_number = None
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
        action_specs = []
        observation_specs = []
        reward_specs = []
        # specs for single agent
        i_action_spec = Categorical(
            n=self.jobs + 1,
            shape=(),
            dtype=torch.int64,
            device="cpu",
        )
        i_observation_spec = Bounded(
            low=False,
            high=True,
            shape=(self.jobs + 1,),
            dtype=torch.bool,
            device="cpu",
        )
        i_reward_spec = Unbounded(shape=(1,))
        # each agent represents a machine
        for i in range(self.machines):
            action_specs.append(i_action_spec)
            observation_specs.append(i_observation_spec)
            reward_specs.append(i_reward_spec)
        # observation
        self.observation_spec = Composite(
            agents = Composite(
                action_mask = torch.stack(observation_specs, dim=0),
                shape=(self.machines,),
            ),
            real_obs = Unbounded(
                shape=(self.jobs, 7),
                dtype=torch.float32,
                device="cpu",
            ),
            shape=(),
        )
        self.state_spec = self.observation_spec.clone()
        # action
        self.action_spec = Composite(
            agents = Composite(
                action = torch.stack(action_specs, dim=0),
                shape=(self.machines,)
            ),
            shape=(),
        )
        # reward
        self.reward_spec = Composite(
            agents = Composite(
                reward = torch.stack(reward_specs, dim=0),
                shape=(self.machines,),
            ),
            shape=(),
        )
        # done
        self.done_spec = Categorical(
            n=2,
            shape=torch.Size((1,)),
            dtype=torch.bool,
        )
    
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
        
        self.machines_legal = np.zeros(self.machines, dtype=bool)
        self.machines_legal_number = 0
        self.machines_time_until_free = np.zeros(self.machines, dtype=np.int32)
        
        self.actions_illegal = np.zeros((self.machines, self.jobs), dtype=bool)
        self.actions_illegal_no_operations = np.zeros(self.jobs, dtype=bool)
        self.actions_legal = np.zeros((self.machines, self.jobs + 1), dtype=bool)
        self.actions_legal_number = np.zeros(self.machines, dtype=np.int32)
        for job in range(self.jobs):
            needed_machine = self.instance_matrix[job][0][0]
            self.jobs_needed_machine[job] = needed_machine
            if not self.machines_legal[needed_machine]:
                self.machines_legal[needed_machine] = True
                self.machines_legal_number += 1
            # legal action section
            self.actions_legal[needed_machine][job] = True
            self.actions_legal_number[needed_machine] += 1
        # WRITE IF NO ACTIONS ARE LEGAL (NO JOBS)
        
        self.state = np.zeros((self.jobs, 7), dtype=np.float32)
        self.solution = np.full((self.jobs, self.machines,), -1, dtype=int)
        
        return self._get_state()
    
    def _step(self, tensordict):
        # each agent has it's own reward, initially zero
        rewards = np.zeros(self.machines, dtype=np.float32)
        actions = tensordict["agents", "action"]
        # each agent (machine) chose an action
        for machine in range(self.machines):
            reward = 0
            action = actions[machine].item()
            # not idle action case
            if action != self.jobs:
                current_time_step_job = self.jobs_todo_time_step[action]
                time_needed = self.instance_matrix[action][current_time_step_job][1]
                reward += time_needed
                self.machines_time_until_free[machine] = time_needed
                self.jobs_time_until_complete_current_operation[action] = time_needed
                # state[1]
                self.state[action][1] = time_needed / self.max_time_operation
                to_add_time_step = self.current_time_step + time_needed
                if to_add_time_step not in self.next_time_step:
                    index = bisect.bisect_left(self.next_time_step, to_add_time_step)
                    self.next_time_step.insert(index, to_add_time_step)
                    self.jobs_next.insert(index, action)
                self.solution[action][current_time_step_job] = self.current_time_step
                for job in range(self.jobs):
                    if (self.jobs_needed_machine[job] == machine
                        and self.actions_legal[machine][job]):
                        self.actions_legal[machine][job] = False
                        self.actions_legal_number[machine] -= 1
                self.machines_legal_number -= 1
                self.machines_legal[machine] = False
                for job in range(self.jobs):
                    if self.actions_illegal[machine][job]:
                        self.actions_illegal_no_operations[job] = False
                        self.actions_illegal[machine][job] = False
            rewards[machine] = self._scale_reward(reward)
        while self.machines_legal_number == 0 and len(self.next_time_step) > 0:
            self._time_step() # UNFINISHED
        state = self._get_state()
        out = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "action_mask": state["agents", "action_mask"],
                        "reward": rewards,
                    },
                    batch_size=[self.machines,],
                ),
                "real_obs": state["real_obs"],
                "done": self._is_done(),
            },
            tensordict.shape,
        )
        return out
        
    def _time_step(self):
        hole_planning = 0
        next_time_step_to_pick = self.next_time_step.pop(0)
        self.jobs_next.pop(0)
        difference = next_time_step_to_pick - self.current_time_step
        self.current_time_step = next_time_step_to_pick
        # JOB WORK
        for job in range(self.jobs):
            # time left to complete current job's operation
            left_time = self.jobs_time_until_complete_current_operation[job]
            # if current job's operation unfinished
            if left_time > 0:
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
                # if during current time step current job's operation is completed
                if self.jobs_time_until_complete_current_operation[job] == 0:
                    self.jobs_total_idle_time[job] += difference - left_time
                    self.state[job][6] = self.jobs_total_idle_time[job] / self.operations_total_time
                    self.jobs_idle_time_last_operation[job] = difference - left_time
                    self.state[job][5] = self.jobs_idle_time_last_operation[job] / self.operations_total_time
                    self.jobs_todo_time_step[job] += 1
                    self.state[job][2] = self.jobs_todo_time_step[job] / self.machines
                    # if finished operation is not last
                    if self.jobs_todo_time_step[job] < self.machines:
                        self.jobs_needed_machine[job] = self.instance_matrix[job][self.jobs_todo_time_step[job]][0]
                        self.state[job][4] = (
                            max(0, self.machines_time_until_free[self.jobs_needed_machine[job]] - difference) / self.max_time_operation
                        )
                    else:
                        self.jobs_needed_machine[job] = -1
                        self.state[job][4] = 1.0
                        for machine in range(self.machines):
                            if self.actions_legal[machine][job]:
                                self.actions_legal[machine][job] = False
                                self.actions_legal_number[machine] -= 1
            # if current job not in work and still not completed
            elif self.jobs_todo_time_step[job] < self.machines:
                self.jobs_total_idle_time[job] += difference
                self.jobs_idle_time_last_operation[job] += difference
                self.state[job][5] = self.jobs_idle_time_last_operation[job] / self.operations_total_time
                self.state[job][6] = self.jobs_total_idle_time[job] / self.operations_total_time
        # MACHINE WORK
        for machine in range(self.machines):
            # if machine will be free sooner than current time step ends
            if self.machines_time_until_free[machine] < difference:
                empty = difference - self.machines_time_until_free[machine]
                hole_planning += empty
            self.machines_time_until_free[machine] = max(0, self.machines_time_until_free[machine] - difference)
            # if machine will be free by the end of current time step
            if self.machines_time_until_free[machine] == 0:
                for job in range(self.jobs):
                    if (self.jobs_needed_machine[job] == machine
                        and not self.actions_legal[machine][job]
                        and not self.actions_illegal[machine][job]):
                        self.actions_legal[machine][self.jobs] = False
                        self.actions_legal[machine][job] = True
                        self.actions_legal_number[machine] += 1
                        if not self.machines_legal[machine]:
                            self.machines_legal[machine] = True
                            self.machines_legal_number += 1
        return hole_planning
    
    def _scale_reward(self, reward):
        return np.float32(reward / self.max_time_operation)
    
    def _is_done(self):
        if np.all(self.actions_legal[:, self.jobs]):
            self.last_time_step = self.current_time_step
            return True
        return False
    
    def _get_state(self):
        for job in range(self.jobs):
            self.state[job][0] = True if self.state[job][1] == 0 and self.state[job][2] < 1.0 else False
        for machine in range(self.machines):
            # if machine has no jobs to work with,
            # make sure to let it do nothing
            if not np.any(self.actions_legal[machine]):
                self.actions_legal[machine][self.jobs] = True
        out = TensorDict(
            {
                "real_obs": self.state,
                "agents": TensorDict(
                    {
                        "action_mask": self.actions_legal,
                    },
                    batch_size=[self.machines,],
                ),
            },
            [],
        )
        return out
    
    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng

# from torchrl.envs import (
#     CatTensors,
#     EnvBase,
#     Transform,
#     TransformedEnv,
#     UnsqueezeTransform,
#     ActionMask,
#     RewardSum,
# )
# from torchrl.envs.transforms.transforms import _apply_to_composite
# from torchrl.envs.utils import check_env_specs, step_mdp

# base_env = MAJSPEnv()
# env = TransformedEnv(
#     base_env,
#     ActionMask(action_key=base_env.action_key, mask_key=("agents", "action_mask")),
# )
# reward_sum = RewardSum(in_keys=env.reward_keys, out_keys=[("agents", "episode_reward")])
# env.append_transform(reward_sum)

# state = env.reset()
# done = False
# while not done:
#     step = env.rand_step(state)
#     print(step["agents", "action"])
#     print(step["next", "agents", "reward"])
#     print(step["agents", "episode_reward"])
#     state = step_mdp(step)
#     done = state["done"]
#     print(done)
