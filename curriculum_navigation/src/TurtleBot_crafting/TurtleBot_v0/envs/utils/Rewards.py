from enum import Enum


class Rewards(Enum):
    REWARD_DONE=1000,
    REWARD_BREAK=50,
    REWARD_STEP=-1,
    REWARD_HIT_WALL=-10,

