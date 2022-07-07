## Environment

If you want to use our environment for your research, you can check this README file.

## Try our environment with random action

We provide several examples in the `example` directory to try our environments, including relocate, pour, and place
inside. It is very simple to use the code, take pour as example:

```python
import numpy as np

from hand_imitation.env.environments.mug_pour_water_env import WaterPouringEnv

if __name__ == '__main__':
    # Environment parameters
    tank_size = (0.15, 0.15, 0.06)  # the size of water tank container
    mug_init_offset = (0.22, 0)  # the initial position of mug
    tank_init_pos = (-0.08, -0.1)  # the initial position of the water tank container

    # Geom parameters for the MuJoCo mug object
    # Please refer to http://www.mujoco.org/book/XMLreference.html#geom for more details
    geom_params = dict(condim="4", margin="0.003")

    # Construct env
    env = WaterPouringEnv(has_renderer=True, tank_size=tank_size, mug_init_offset=mug_init_offset,
                          tank_init_pos=tank_init_pos, **geom_params)
    # or you can simply use the default parameters with:
    # env = WaterPouringEnv(has_renderer=True)
    env.seed(0)

    # Action spec
    obs = env.reset()
    low, high = env.action_spec

    # Visualization
    for i in range(500):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()
```
