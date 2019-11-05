# -*- coding: utf-8 -*-
# @Author: fyr91
# @Date:   2019-10-04 15:55:09
# @Last Modified by:   fyr91
# @Last Modified time: 2019-11-05 22:11:42
import gym
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from gym import spaces
import logging
import numpy as np

logger = logging.getLogger(__name__)
NO_OP = actions.RAW_FUNCTIONS.no_op.id
MOVE_PT = actions.RAW_FUNCTIONS.Move_pt.id

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
        'realtime': False
    }


    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.env = None
        self.marines = []
        self.banelings = []
        self.zerglings = []
        # 000 idle
        # 1[0~1][0-8] move 0-8 up
        # 1[1~2][0-8] move 0-8 down
        # 1[2~3][0-8] move 0-8 left
        # 1[3~4][0-8] move 0-8 right
        # 2[0-8][9-18] use 0-8 to attack 0-9
        self.action_space = spaces.Discrete(123)
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


    # def init_env(self):
    #     args = {**self.default_settings, **self.kwargs}
    #     self.env =  sc2_env.SC2Env(**args)


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
            self.marines.append(m)
            obs[i] = np.array([1, m.x, m.y, m[2]])

        for i, b in enumerate(banelings):
            self.banelings.append(b)
            obs[i+9] = np.array([1, b.x, b.y, b[2]])

        for i, z in enumerate(zerglings):
            self.zerglings.append(z)
            obs[i+13] = np.array([1, z.x, z.y, z[2]])

        return obs


    def take_action(self, action):
        if action == 0:
            action_mapped = actions.RAW_FUNCTIONS.no_op()
        elif action<=32:
            derived_action = np.floor((action-1)/8)
            idx = (action-1)%8
            if derived_action == 0:
                action_mapped = self.move_up(idx)
            elif derived_action == 1:
                action_mapped = self.move_down(idx)
            elif derived_action == 2:
                action_mapped = self.move_left(idx)
            else:
                action_mapped = self.move_right(idx)
        else:
            eidx = np.floor((action-33)/9)
            aidx = (action-33)%9
            action_mapped = self.attack(aidx, eidx)

        print(f'{self.steps}: {action_mapped}')
        raw_obs = self.env.step([action_mapped])[0]

        return raw_obs        # try:
        # # except:
        # #     print(f'{self.steps}: no operation (exception)')
        # #     raw_obs = self.env.step([actions.FunctionCall(NO_OP,[])])[0]

        # return raw_obs



    def move_up(self, idx):
        idx = np.floor(idx)
        try:
            selected = self.marines[idx]
            new_pos = [selected.x, selected.y-2]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
            # return [MOVE_PT, [['now'], [selected.tag], [selected.x, selected.y-2]]]
        except:
            return actions.RAW_FUNCTIONS.no_op()
            # return [NO_OP, []]

        # return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, (selected.x, selected.y-2))
        # return [actions.RAW_FUNCTIONS.no_op.id,[]]


    def move_down(self, idx):
        try:
            selected = self.marines[idx]
            new_pos = [selected.x, selected.y+2]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()
            # return [NO_OP, []]

        # return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, (selected.x, selected.y+2))
        # return [actions.RAW_FUNCTIONS.no_op.id,[]]


    def move_left(self, idx):
        try:
            selected = self.marines[idx]
            new_pos = [selected.x-2, selected.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()
            # return [NO_OP, []]
        # return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, (selected.x-2, selected.y))
        # return [actions.RAW_FUNCTIONS.no_op.id,[]]


    def move_right(self, idx):
        try:
            selected = self.marines[idx]
            new_pos = [selected.x+2, selected.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()
            # return [NO_OP, []]
        # return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, (selected.x+2, selected.y))
        # return [actions.RAW_FUNCTIONS.no_op.id, []]


    def attack(self, aidx, eidx):
        try:
            selected = self.marines[aidx]
            if edix>3:
                # attack zerglines
                target = self.zerglines[eidx-4]
            else:
                target = self.banelings[eidx]
            return actions.RAW_FUNCTIONS.Attack_unit("now", selected.tag, targeted.tag)
        except:
            return actions.RAW_FUNCTIONS.no_op()
            # return [NO_OP, []]
        # return [actions.RAW_FUNCTIONS.no_op.id,[]]


    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env =  sc2_env.SC2Env(**args)


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
