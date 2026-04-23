import os
import hydra
import torch
import utils
from train_spd import make_eval_1_env, make_eval_2_env
from logger import Logger

@hydra.main(config_path='.', config_name='config_spd')
def main(cfg):
    from spd import SPDAgent
    import hydra as h

    eval_1_env = make_eval_1_env(cfg)
    eval_2_env = make_eval_2_env(cfg)

    cfg.agent.obs_shape = eval_1_env.observation_space.shape
    cfg.agent.action_shape = eval_1_env.action_space.shape
    cfg.agent.action_range = [float(eval_1_env.action_space.low.min()),
                               float(eval_1_env.action_space.high.max())]

    agent = h.utils.instantiate(cfg.agent, _recursive_=False)

    model_dir = "C:/Users/tpenu/OneDrive/Documents/GitHub/SPD-VBRL/runs_spd/2026.04.20/123641_/models"
    step = 0  # change to the checkpoint step you want
    agent.load(model_dir, step)

    for env_name, env in [("eval_1", eval_1_env), ("eval_2", eval_2_env)]:
        total_reward = 0
        for ep in range(cfg.num_eval_episodes):
            obs, done, ep_reward = env.reset(), False, 0
            while not done:
                with utils.eval_mode(agent):
                    action = agent.act(obs, sample=False)
                obs, reward, done, _ = env.step(action)
                ep_reward += reward
            total_reward += ep_reward
        print(f"{env_name} avg reward: {total_reward / cfg.num_eval_episodes:.2f}")

if __name__ == '__main__':
    main()
