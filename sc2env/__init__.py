# -*- coding: utf-8 -*-
# @Author: fyr91
# @Date:   2019-10-04 15:55:09
# @Last Modified by:   fyr91
# @Last Modified time: 2019-10-11 14:01:04

from gym.envs.registration import register

register(
    id='defeat-zerglings-banelings-v0',
    entry_point='sc2env.envs:DZBEnv',
)
