import bidexhands as bi
import torch
import yaml
from bidexhands.utils.package_utils import process_sarl
from argparse import Namespace

args = Namespace(
    algo="ppo",
    model_dir="",
    max_iterations=-1
)

env_name = 'ShadowHandBottleCap'
algo = "ppo"
checkpoint_dir = "./logs/ShadowHandBottleCap/ppo/ppo_seed-1/"
config_path = ""
env = bi.make(env_name, algo)
yaml.safe_load(open(checkpoint))

obs = env.reset()
terminated = False

while not terminated:
    act = torch.tensor(env.action_space.sample()).repeat((env.num_envs, 1))
    obs, reward, done, info = env.step(act)
