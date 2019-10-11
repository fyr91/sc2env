# -*- coding: utf-8 -*-
# @Author: fyr91
# @Date:   2019-10-04 15:55:09
# @Last Modified by:   fyr91
# @Last Modified time: 2019-10-11 16:27:08
import gym
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from gym import spaces
import logging
import numpy as np

logger = logging.getLogger(__name__)

class DZBEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    default_settings = {
        'map_name': "DefeatZerglingsAndBanelings",
        'players': [sc2_env.Agent(sc2_env.Race.terran),
                    sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.hard)],
        'agent_interface_format': features.AgentInterfaceFormat(
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64),
        'realtime': True
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.env = None
        self.marines = []
        self.banelings = []
        self.zerglings = []
        # 000 idle
        # 10[0-8] move 0-8 up
        # 11[0-8] move 0-8 down
        # 12[0-8] move 0-8 left
        # 13[0-8] move 0-8 right
        # 2[0-8][9-18] use 0-8 to attack 9-18
        self.action_space = spaces.Box(
                low=np.array([0,0,0]),
                high=np.array([2,8,18]),
                dtype=np.uint8
            )
        # [0: survived, 1: x, 2: y, 3: hp]
        self.observation_space = spaces.Box(
            low=0,
            high=64,
            shape=(19,4),
            dtype=np.uint8
            )
        self.steps = 0
        self.episodes = 0
        self.episode_reward = 0
        self.total_reward = 0


    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env =  sc2_env.SC2Env(**args)


    def step(self, action):
        self.steps += 1
        raw_obs = self.take_action(action)
        reward = raw_obs.reward
        obs = self.get_derived_obs(raw_obs)
        self.episode_reward += reward
        self.total_reward += reward
        # each step will set the dictionary to emtpy
        return obs, reward, raw_obs.last(), {}


    def get_units_by_type(self, obs, unit_type, player_relative=0):
        """
        NONE = 0
        SELF = 1
        ALLY = 2
        NEUTRAL = 3
        ENEMY = 4
        """
        return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == player_relative]


    def get_derived_obs(self, raw_obs):
        obs = np.zeros((19,4), dtype=np.uint8)

        marines = self.get_units_by_type(raw_obs, units.Terran.Marine, 1)
        zerglings = self.get_units_by_type(raw_obs, units.Zerg.Zergling, 4)
        banelings = self.get_units_by_type(raw_obs, units.Zerg.Baneling, 4)

        self.marines = []
        self.banelings = []
        self.zerglings = []

        for i, m in enumerate(marines):
            self.marines.append(m.tag)
            obs[i] = np.array([1, m.x, m.y, m[2]])

        for i, b in enumerate(banelings):
            self.banelings.append(b.tag)
            obs[i+9] = np.array([1, b.x, b.y, b[2]])

        for i, z in enumerate(zerglings):
            self.zerglings.append(z.tag)
            obs[i+13] = np.array([1, z.x, z.y, z[2]])

        return obs


    def take_action(self, action):
        if action[2] >= 9:
            if action[0] == 1 and action[1] == 0:
                action_mapped = self.move_up(action[2])
            elif action[0] == 1 and action[1] == 1:
                action_mapped = self.move_down(action[2])
            elif action[0] == 1 and action[1] == 2:
                action_mapped = self.move_left(action[2])
            elif action[0] == 1 and action[1] == 3:
                action_mapped = self.move_right(action[2])
            elif action[0] == 2 and action[1] < 9:
                action_mapped = self.attact(action[1], action[2])
        else:
            action_mapped = [actions.RAW_FUNCTIONS.no_op.id,[]]

        try:
            raw_obs = self.env.step([actions.FunctionCall(action_mapped[0],action_mapped[1])])[0]
        except:
            raw_obs = self.env.step([actions.FunctionCall(actions.RAW_FUNCTIONS.no_op.id,[])])[0]

        return raw_obs


    def move_up(self, idx):
        # return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, (selected.x, selected.y-2))
        return [actions.RAW_FUNCTIONS.no_op.id,[]]


    def move_down(self, idx):
        # return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, (selected.x, selected.y+2))
        return [actions.RAW_FUNCTIONS.no_op.id,[]]


    def move_left(self, idx):
        # return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, (selected.x-2, selected.y))
        return [actions.RAW_FUNCTIONS.no_op.id,[]]


    def move_right(self, idx):
        # return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, (selected.x+2, selected.y))
        return [actions.RAW_FUNCTIONS.no_op.id,[]]


    def attack(self, attacker, enemy):
        return [actions.RAW_FUNCTIONS.no_op.id,[]]


    def reset(self):
        if self.env is None:
            self.init_env()
        if self.episodes > 0:
            logger.info(f"Episode {self.episodes} gained {self.episode_reward} reward with {self.steps} steps")
            logger.info(f"Average reward per episode: {self.total_reward/self.steps}")

        self.episodes += 1
        self.steps = 0
        self.episode_reward = 0
        self.marines = []
        self.banelings = []
        self.zerglings = []

        logger.info(f"Episode {self.episodes} started")
        raw_obs = self.env.reset()[0]
        return self.get_derived_obs(raw_obs)


    def close(self):
        if self.episode > 0:
            logger.info(f"Episode {self.episodes} gained {self.episode_reward} reward with {self.steps} steps")
            logger.info(f"Average reward per episode: {self.total_reward/self.steps}")
        if self.env is not None:
            self.env.close()
        super().close()


    def render(self, mode='human', close=False):
        pass
