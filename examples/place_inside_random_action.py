import time

import numpy as np

from hand_imitation.env.environments.mug_place_object_env import MugPlaceObjectEnv

if __name__ == '__main__':
    # For performance consideration, please set load_eval_mesh=False for training, and only use True for eval
    env = MugPlaceObjectEnv(has_renderer=True, friction=(1, 0.5, 0.01), object_scale=0.8, mug_scale=1.5,
                            large_force=True, load_eval_mesh=True)
    obs = env.reset()
    low, high = env.action_spec

    # Loop visualization
    tic = time.time()
    for i in range(200):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()

    # since it is a random policy, you will just get 0 score
    print(f"Volume percentage: {env.compute_intersection_rate()}")
