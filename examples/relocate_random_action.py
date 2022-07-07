import numpy as np

from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate

if __name__ == '__main__':
    object_names = ["mug", "mustard_bottle", "tomato_soup_can", "large_clamp", "sugar_box", "potted_meat_can",
                    "foam_brick"]

    for object_name in object_names:
        env = YCBRelocate(has_renderer=True, object_name=object_name, friction=(1, 0.5, 0.01),
                          object_scale=0.8, version="v2")
        env.seed(0)
        env.reset()
        low, high = env.action_spec

        # Loop visualization
        for i in range(200):
            action = np.random.uniform(low, high)
            obs, reward, done, _ = env.step(action)
            env.render()
