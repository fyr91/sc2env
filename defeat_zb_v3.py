# -*- coding: utf-8 -*-
# @Author: fyr91
# @Date:   2019-10-04 15:57:18
# @Last Modified by:   fyr91
# @Last Modified time: 2019-10-04 16:08:49
import random
import numpy as np
import pandas as pd
from absl import app
from scipy.spatial.distance import cdist
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, q_table_file=None):
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        if q_table_file:
            self.q_table = pd.read_pickle(q_table_file)
        else:
            self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)


    def choose_action(self, observation, e_greedy=0.9):
        self.check_state_exist(observation)
        if np.random.uniform() < e_greedy:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.reward_decay * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions),
                index=self.q_table.columns,
                name=state))

    def save(self, episodes):
        self.q_table.to_pickle(f"q_tables/q_table_{episodes}.pkl")

class RawAgent(base_agent.BaseAgent):
    actions = [
        "retreat_weakest",
        "retreat_closest",
        # "move_up",
        # "move_down",
        # "move_left",
        # "move_right",
        # "attack_closest_zerglings",
        "attack_weakest_zerglings",
        # "attack_closest_banelings",
        "attack_weakest_banelings",
        "do_nothing"
    ]

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

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

    def get_enemy_units(self, obs):
        """
        NONE = 0
        SELF = 1
        ALLY = 2
        NEUTRAL = 3
        ENEMY = 4
        """
        return [unit for unit in obs.observation.raw_units if unit.alliance == 4]

    def step(self, obs):
        super(RawAgent, self).step(obs)

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def retreat_weakest(self, obs):
        # retreat the weakest
        marines = self.get_units_by_type(obs, units.Terran.Marine, 1)
        enemies = self.get_enemy_units(obs)
        if len(marines) > 0 and len(enemies) > 0:
            selected = marines[np.argmin(list(zip(*marines))[2])]
            distances = self.get_distances(obs, enemies, (selected.x, selected.y))
            closest = enemies[np.argmin(distances)]
            relative_pos = [closest.x - selected.x, closest.y - selected.y]
            if relative_pos[0] >= 0:
                if relative_pos[1] >= 0:
                    new_pos = (selected.x-1, selected.y-1)
                else:
                    new_pos = (selected.x-1, selected.y+1)
            else:
                if relative_pos[1] >= 0:
                    new_pos = (selected.x+1, selected.y-1)
                else:
                    new_pos = (selected.x+1, selected.y+1)
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def retreat_closest(self, obs):
        # retreat closest
        marines = self.get_units_by_type(obs, units.Terran.Marine, 1)
        enemies = self.get_enemy_units(obs)
        if len(marines) > 0 and len(enemies) > 0:
            closest_pairs = []
            closest_distances = []
            for i,m in enumerate(marines):
                distances = self.get_distances(obs, enemies, (m.x, m.y))
                j = np.argmin(distances)
                closest_pairs.append([i,j])
                closest_distances.append(distances[j])
            pair = closest_pairs[np.argmin(closest_distances)]
            selected = marines[pair[0]]
            closest = enemies[pair[1]]
            relative_pos = [closest.x - selected.x, closest.y - selected.y]
            if relative_pos[0] >= 0:
                if relative_pos[1] >= 0:
                    new_pos = (selected.x-1, selected.y-1)
                else:
                    new_pos = (selected.x-1, selected.y+1)
            else:
                if relative_pos[1] >= 0:
                    new_pos = (selected.x+1, selected.y-1)
                else:
                    new_pos = (selected.x+1, selected.y+1)
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        else:
            return actions.RAW_FUNCTIONS.no_op()


    def move_up(self, obs):
        # control units with least hp
        marines = self.get_units_by_type(obs, units.Terran.Marine, 1)
        if len(marines) > 0:
            selected = marines[np.argmin(list(zip(*marines))[2])]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, (selected.x, selected.y-2))
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def move_down(self, obs):
        # control units with least hp
        marines = self.get_units_by_type(obs, units.Terran.Marine, 1)
        if len(marines) > 0:
            selected = marines[np.argmin(list(zip(*marines))[2])]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, (selected.x, selected.y+2))
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def move_left(self, obs):
        # control units with least hp
        marines = self.get_units_by_type(obs, units.Terran.Marine, 1)
        if len(marines) > 0:
            selected = marines[np.argmin(list(zip(*marines))[2])]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, (selected.x-2, selected.y))
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def move_right(self, obs):
        # control units with least hp
        marines = self.get_units_by_type(obs, units.Terran.Marine, 1)
        if len(marines) > 0:
            selected = marines[np.argmin(list(zip(*marines))[2])]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, (selected.x+2, selected.y))
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def attack_closest_zerglings(self, obs):
        # get the strongest marine to attack the closest
        marines = self.get_units_by_type(obs, units.Terran.Marine, 1)
        zerglings = self.get_units_by_type(obs, units.Zerg.Zergling, 4)
        if len(zerglings) > 0 and len(marines) > 0:
            selected = marines[np.argmax(list(zip(*marines))[2])]
            distances = self.get_distances(obs, zerglings, (selected.x, selected.y))
            targeted = zerglings[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Attack_unit("now", selected.tag, targeted.tag)
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def attack_weakest_zerglings(self, obs):
        # get the nonweak one to attack the weakest
        marines = self.get_units_by_type(obs, units.Terran.Marine, 1)
        zerglings = self.get_units_by_type(obs, units.Zerg.Zergling, 4)
        if len(zerglings) > 0 and len(marines) > 0:
            targeted = zerglings[np.argmin(list(zip(*zerglings))[2])]
            if len(marines) == 1:
                selected = marines[0]
            else:
                weakeast = marines[np.argmin(list(zip(*marines))[2])]
                nonweak = [m for m in marines if m.tag != weakeast.tag]
                selected = random.choice(nonweak)
            return actions.RAW_FUNCTIONS.Attack_unit("now", selected.tag, targeted.tag)
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def attack_closest_banelings(self, obs):
        # get the strongest one to attack
        marines = self.get_units_by_type(obs, units.Terran.Marine, 1)
        banelings = self.get_units_by_type(obs, units.Zerg.Baneling, 4)
        if len(banelings) > 0 and len(marines) > 0:
            selected = marines[np.argmax(list(zip(*marines))[2])]
            distances = self.get_distances(obs, banelings, (selected.x, selected.y))
            targeted = banelings[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Attack_unit("now", selected.tag, targeted.tag)
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def attack_weakest_banelings(self, obs):
        # get the closest one to attack the weakest
        marines = self.get_units_by_type(obs, units.Terran.Marine, 1)
        banelings = self.get_units_by_type(obs, units.Zerg.Baneling, 4)
        if len(banelings) > 0 and len(marines) > 0:
            targeted = banelings[np.argmin(list(zip(*banelings))[2])]
            if len(marines) == 1:
                selected = marines[0]
            else:
                weakeast = marines[np.argmin(list(zip(*marines))[2])]
                nonweak = [m for m in marines if m.tag != weakeast.tag]
                selected = random.choice(nonweak)
            return actions.RAW_FUNCTIONS.Attack_unit("now", selected.tag, targeted.tag)
        else:
            return actions.RAW_FUNCTIONS.no_op()


class SmartAgent(RawAgent):
    def __init__(self, learning=True, q_table_file=None):
        super(RawAgent, self).__init__()
        self.q_table = QLearningTable(self.actions, q_table_file=q_table_file)
        self.learning = learning
        self.num_marine = 9
        self.num_zergling = 6
        self.num_baneling = 4
        self.MAX_MARINE_HEALTH = 45
        self.MAX_ZERGLING_HEALTH = 35
        self.MAX_BANELING_HEALTH = 30
        self.new_game()

    def reset(self):
        super(SmartAgent, self).reset()
        self.new_game()

    def new_game(self):
        self.previous_state = None
        self.previous_action = None

    def discretize_health(self, h, max_health, num):
        thresholds = np.linspace(start=0, stop=max_health, num=num)
        # print(thresholds)
        for i,th in enumerate(thresholds):
            if h <= th:
                return i
            else:
                continue


    def get_state(self, obs):
        marines = self.get_units_by_type(obs, units.Terran.Marine, 1)
        zerglings = self.get_units_by_type(obs, units.Zerg.Zergling, 4)
        banelings = self.get_units_by_type(obs, units.Zerg.Baneling, 4)
        # marines_health = [0]*self.num_marine
        # zerglings_health = [0]*self.num_zergling
        # banelings_health = [0]*self.num_baneling
        # if len(marines) > 0:
        #     marines_health += list(zip(*marines))[2]、
        # if len(zerglings) > 0:
        #     zerglings_health += list(zip(*zerglings))[2]
        # if len(banelings) > 0:
        #     banelings_health += list(zip(*banelings))[2]
        # state = [len(marines), len(zerglings), len(banelings)]
        # state.extend(marines_health)
        # state.extend(zerglings_health)
        # state.extend(banelings_health)
        # get the least health for each kinds
        if len(marines) > 0:
            least_marine_health = int(np.min(list(zip(*marines))[2]))
        else:
            least_marine_health = 0

        if len(zerglings) > 0:
            least_zergling_health = int(np.min(list(zip(*zerglings))[2]))
        else:
            least_zergling_health = 0

        if len(banelings) > 0:
            least_baneling_health = int(np.min(list(zip(*banelings))[2]))
        else:
            least_baneling_health = 0

        try:
            marines_pos = [(m.x, m.y) for m in marines]
            zerglings_pos = [(z.x, z.y) for z in zerglings]
            cdist1 = int(cdist(marines_pos, zerglings_pos).min())
        except:
            cdist1 = 999

        try:
            marines_pos = [(m.x, m.y) for m in marines]
            banelings_pos = [(b.x, b.y) for z in banelings]
            cdist2 = int(cdist(marines_pos, banelings_pos).min())
        except:
            cdist2 = 999

        # print(type(least_marine_health))

        state = [len(marines), len(zerglings), len(banelings),
                 self.discretize_health(least_marine_health, self.MAX_MARINE_HEALTH, 5),
                 self.discretize_health(least_zergling_health, self.MAX_ZERGLING_HEALTH, 5),
                 self.discretize_health(least_baneling_health, self.MAX_BANELING_HEALTH, 5),
                 cdist1,
                 cdist2]
        # print(state)
        return state

    def step(self, obs):
        super(SmartAgent, self).step(obs)
        state = str(self.get_state(obs))
        action = self.q_table.choose_action(state)
        if self.learning:
            if self.previous_action is not None:
                self.q_table.learn(self.previous_state, self.previous_action, obs.reward, 'terminal' if obs.last() else state)
            self.previous_state = state
            self.previous_action = action
            if self.episodes % 1000 == 0:
                self.q_table.save(self.episodes)
        return getattr(self, action)(obs)

def main(unused_argv):
    # agent = SmartAgent()
    agent = SmartAgent(learning=False, q_table_file="selected_q_table/v3_90000.pkl")
    try:
        while True:
            with sc2_env.SC2Env(
                map_name="DefeatZerglingsAndBanelings",
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         sc2_env.Bot(sc2_env.Race.zerg,
                                     sc2_env.Difficulty.hard)],
                agent_interface_format=features.AgentInterfaceFormat(
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64,
                ),
                realtime = True,
                # step_mul = 4
            ) as env:
                # run_loop.run_loop([agent], env, max_episodes=1000)
                run_loop.run_loop([agent], env)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)
