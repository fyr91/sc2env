import gym
from pysc2.env import sc2_env
from pysc2.lib import actions
from gym import spaces
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SC2GymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.kwargs = kwargs
        self.env = None
        self.action_space = spaces.Discrete(5)
        self.steps = 0
        self.episodes = 0
        self.episode_reward = 0
        self.total_reward = 0

    def init_env(self):
        self.env =  sc2_env.SC2Env(**self._kwargs)

    def step(self, action):
        self.steps+=1
        obs = self.env.step(actions.RAW_FUNCTIONS.no_op())
        reward = obs.reward
        self.episode_reward += reward
        self.total_reward += reward
        return obs, reward, obs.last(), {}

    def reset(self):
        if self.env is None:
            self.init_env()
        if self.episodes > 0:
            logger.info(f"Episode {self.episodes} gained {self.episode_reward} reward with {self.steps} steps")
            logger.info(f"Average reward per episode: {self.total_reward/self.steps}")

        self.episodes += 1
        self.steps = 0
        self.episode_reward = 0
        logger.info(f"Episode {self.episodes} started")
        obs = self.env.reset()[0]
        return obs

    def close(self):
        if self.episode > 0:
            logger.info(f"Episode {self.episodes} gained {self.episode_reward} reward with {self.steps} steps")
            logger.info(f"Average reward per episode: {self.total_reward/self.steps}")
        if self.env is not None:
            self.env.close()
        super().close()

    def render(self, mode='human', close=False):
        pass
