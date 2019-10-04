# -*- coding: utf-8 -*-
# @Author: fyr91
# @Date:   2019-10-04 15:55:09
# @Last Modified by:   fyr91
# @Last Modified time: 2019-10-04 16:35:27

from gym.envs.registration import register

register(
    id='defeat-zerglings-banelings-v0',
    entry_point='sc2env.envs:DZBEnv',
    kwargs={
        'map_name': 'DefeatZerglingsAndBanelings'
    }
)

register(
    id='sc2-gym-v0',
    entry_point='sc2env.envs:SC2GymEnv',
    kwargs={}
)
