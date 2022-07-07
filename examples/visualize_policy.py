import click
import pickle

DESC = '''
Helper script to visualize learning.\n
USAGE:\n
    Visualizes learning on the env\n
    $ python visualize_policy --env_name relocate --object_name potted_meat_can \n
'''
from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate
from hand_imitation.env.environments.mug_pour_water_env import WaterPouringEnv
from hand_imitation.env.environments.mug_place_object_env import MugPlaceObjectEnv


# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required=True)
@click.option('--object_name', type=str, help='environment to load', required=False, default=None)
def main(env_name, object_name):
    friction = (1, 0.5, 0.01)
    if env_name == "relocate":
        if object_name is None:
            raise ValueError("For relocate task, object name is needed.")
        e = YCBRelocate(has_renderer=True, object_name=object_name, friction=friction, object_scale=0.8,
                        solref="-6000 -300", randomness_scale=0.25)
        T = 100
    elif env_name == "pour":
        e = WaterPouringEnv(has_renderer=True, scale=1.0, tank_size=(0.15, 0.15, 0.08))
        T = 200
    elif env_name == "place_inside":
        e = MugPlaceObjectEnv(has_renderer=True, object_scale=0.8, mug_scale=1.5)
        T = 200
    else:
        raise NotImplementedError

    if env_name == "relocate":
        policy = f"../pretrained_model/{env_name}-{object_name}.pickle"
    else:
        policy = f"../pretrained_model/{env_name}.pickle"
    pi = pickle.load(open(policy, 'rb'))
    for _ in range(100):
        state, done = e.reset(), False
        step = 0
        reward_sum = 0
        while True:
            step += 1
            if step >= T:
                break
            action = pi.get_action(state)[1]['evaluation']
            state, reward, done, _ = e.step(action)
            for _ in range(2):
                e.render()
            reward_sum += reward
        print(f'total reward {reward_sum}')


if __name__ == '__main__':
    main()
