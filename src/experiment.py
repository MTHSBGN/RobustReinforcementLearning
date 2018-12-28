import numpy as np

from agents import Agent
from environments import get_env_creator
from evaluator import Evaluator
from vec_env import SubprocVecEnv


class Experiment:
    def __init__(self, agent: Agent, env_name, num_envs):
        self.agent = agent

        env_creator = get_env_creator(env_name)
        self.envs = SubprocVecEnv([env_creator for _ in range(num_envs)])
        self.evaluator = Evaluator(agent, env_creator())

        self.results = []

    def run(self, num_steps=5, max_frames=100000):
        state = self.envs.reset()
        frame_idx = 0
        while frame_idx < max_frames:
            observations = []
            actions = []
            rewards = []
            masks = []

            for _ in range(num_steps):
                observations.append(state)
                action = self.agent.select_actions(state)
                actions.append(action)
                next_state, reward, done, _ = self.envs.step(action)
                rewards.append(reward)
                masks.append(1 - done)

                state = next_state
                frame_idx += 1

            last_observation = state
            self.agent.step(np.vstack(observations), np.concatenate(actions), rewards, masks, last_observation)

            if frame_idx % 1000 == 0:
                stats = self.evaluator.evaluate()
                print(stats)
                self.results.append(stats)

                if stats["mean_reward"] >= 195:
                    break

    def save_results(self, filename):
        pass
