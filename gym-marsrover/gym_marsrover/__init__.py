from gym.envs.registration import register

register(
    id='marsrover-v0',
    entry_point='gym_marsrover.envs:MarsRover',
)