import torch
import h5py
import os
from tqdm import tqdm
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler


class RolloutStorage:

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        states_shape,
        actions_shape,
        device="cpu",
        sampler="sequential",
    ):

        self.device = device
        self.sampler = sampler

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.states = torch.zeros(num_transitions_per_env, num_envs, *states_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(self, observations, states, actions, rewards, dones, values, actions_log_prob, mu, sigma):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.values[self.step].copy_(values)
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    def clear_vec(self, dones):
        self.step_vec[dones] = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        if self.sampler == "sequential":
            # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
            # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
            subset = SequentialSampler(range(batch_size))
        elif self.sampler == "random":
            subset = SubsetRandomSampler(range(batch_size))

        batch = BatchSampler(subset, mini_batch_size, drop_last=True)
        return batch


class RolloutDataset(RolloutStorage):
    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        actions_shape,
        save_path,
        num_rollouts,
        device="cpu",
        sampler="sequential",
        success_reward_filter=None,
    ):

        self.device = device
        self.sampler = sampler

        # Core
        self.observations = {key: torch.zeros(num_transitions_per_env, num_envs, *obs_shape.shape, device=self.device) for key, obs_shape in obs_shape.items()}
        # self.states = torch.zeros(num_transitions_per_env, num_envs, *states_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0
        self.step_vec = torch.zeros(num_envs, device=self.device, dtype=int)
        self.success_reward_filter = success_reward_filter  # if float, only save demos where the last reward is greater than this
        self.save_path = save_path
        self.num_rollouts = num_rollouts
        if self.num_rollouts > 0:
            self.pbar = tqdm(total=self.num_rollouts, desc="Saving demos")
        if os.path.exists(self.save_path):
            with h5py.File(self.save_path, "r") as f:
                if "data" in f:
                    self.pbar = tqdm(total=self.num_rollouts, desc="Saving demos", initial=len(f["data"]))


    def save_hdf5(self):
        with h5py.File(self.save_path, "a") as f:
            # Store each episode as a separate demo
            if "data" not in f:
                grp = f.create_group("data")
            else:
                grp = f["data"]

            # Get the done values of all envs at the current step
            done_eps = torch.gather(self.dones, 0, (self.step_vec - 1).view(-1, 1, 1)).squeeze()

            # Get the done indices of all envs at the current step
            done_indices = torch.nonzero(done_eps, as_tuple=True)[0]

            ep = len(grp)

            for env_idx in done_indices:
                ep = len(grp)
                if ep >= self.num_rollouts:
                    break
                # If done and not a success, continue
                # check if last reward is greater than self.success_reward_filter
                if self.success_reward_filter and self.rewards[-1, env_idx] < self.success_reward_filter:
                    continue
                grp_ep = grp.create_group(f"demo_{ep}")
                grp_ep.attrs["num_samples"] = int(self.step_vec[env_idx]) - 1
                grp_obs = grp_ep.create_group("obs")
                for key in self.observations.keys():
                    data = self.observations[key][:int(self.step_vec[env_idx]) - 1, env_idx].cpu().numpy()
                    grp_obs.create_dataset(key, data=data, compression="gzip")

                for key in ["actions", "rewards", "dones"]:
                    data = getattr(self, key)[:int(self.step_vec[env_idx]) - 1, env_idx].cpu().numpy()
                    grp_ep.create_dataset(key, data=data, compression="gzip")
                self.pbar.update(1)

            # Reset the step for done episodes
            self.clear_vec(done_eps)
        return ep

    def add_rollout_transitions(self, observations, actions, rewards, dones):
        for env_idx in range(self.num_envs):
            step = int(self.step_vec[env_idx])
            if step >= self.num_transitions_per_env:
                raise AssertionError(f"Rollout buffer overflow at index {env_idx}")

            for key in self.observations.keys():
                self.observations[key][step, env_idx].copy_(observations[key][env_idx])
            self.actions[step, env_idx].copy_(actions[env_idx])
            self.rewards[step, env_idx].copy_(rewards[env_idx].view(-1))
            self.dones[step, env_idx].copy_(dones[env_idx].view(-1))

            self.step_vec[env_idx] += 1

