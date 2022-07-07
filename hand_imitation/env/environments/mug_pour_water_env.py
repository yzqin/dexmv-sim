import xml.etree.ElementTree as ET

import numpy as np
import transforms3d

from hand_imitation.env.environments.base import MujocoEnv
from hand_imitation.env.models import TableArena
from hand_imitation.env.models.base import MujocoXML
from hand_imitation.env.models.objects import YCB_SIZE
from hand_imitation.env.utils.mjcf_utils import xml_path_completion, array_to_string, string_to_array, find_elements
from hand_imitation.env.utils.random import np_random
from hand_imitation.misc.pose_utils import inverse_pose


class WaterPouringEnv(MujocoEnv):
    def __init__(self, has_renderer, tank_size=(0.15, 0.15, 0.06), mug_init_offset=(0.22, 0),
                 tank_init_pos=(-0.08, -0.1), render_gpu_device_id=-1, randomness_scale=1,
                 **kwargs):
        self.np_random = None
        self.seed()
        self.__additional_kwargs = kwargs

        self.object_name = "mug"
        self.mug_init_offset = np.array(mug_init_offset)
        self.num_particles = 0
        self.tank_size = np.array(tank_size)
        self.tank_pos = np.array(tank_init_pos)
        self.randomness_scale = randomness_scale

        super().__init__(has_renderer=has_renderer, has_offscreen_renderer=False, render_camera=None,
                         render_gpu_device_id=render_gpu_device_id, control_freq=100, horizon=200, ignore_done=True,
                         hard_reset=False)

        # Setup action range
        self.act_mid = np.mean(self.mjpy_model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.mjpy_model.actuator_ctrlrange[:, 1] - self.mjpy_model.actuator_ctrlrange[:, 0])

    def _pre_action(self, action, policy_step=False):
        action = np.clip(action, -1.0, 1.0)
        action = self.act_mid + action * self.act_rng  # mean center and scale
        self.sim.data.ctrl[:] = action

    def _initialize_sim(self, xml_string=None):
        super()._initialize_sim(xml_string)
        self.original_particle_pos = self.mjpy_model.body_pos[-self.num_particles:, :]

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.forward()
        self.sim.set_state(self.sim_state_initial)
        mug_offset = np.array(
            [self.np_random.uniform(low=-0.06, high=0.04),
             self.np_random.uniform(low=0.0, high=0.1)]) * self.randomness_scale
        self.sim.data.qpos[
        -self.num_particles * 3 - 7: -self.num_particles * 3 - 5] = mug_offset + self.mug_init_offset + self.tank_pos
        particle_pos = np.zeros_like(self.original_particle_pos) + np.array([[mug_offset[0], mug_offset[1], 0]])
        self.sim.data.qpos[-self.num_particles * 3:] = particle_pos.ravel()
        self.sim.forward()
        return

    def _get_observations(self):
        qp = self.data.qpos.ravel()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_sid].ravel()
        obj_quat = self.data.body_xquat[self.obj_bid].ravel()
        return np.concatenate([qp[:30], palm_pos - obj_pos, palm_pos - target_pos, obj_pos - target_pos, obj_quat])

    def reward(self, action):
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_sid].ravel()
        is_contact = self.check_contact(self.body_geom_names, self.robot_geom_names)

        # Punish particles out of mug
        out_of_mug_bool = ~self.check_in_mug_particles()
        out_of_container_bool = ~self.check_above_particle()
        dropping_bool = np.logical_and(out_of_mug_bool, out_of_container_bool)
        dropping_num = np.sum(dropping_bool)
        dropping_ratio = dropping_num / self.num_particles

        # Relocation Reward
        reward = -0.1 * np.linalg.norm(palm_pos - obj_pos)  # take hand to object
        if is_contact:
            reward += 0.1
            lift = max(min(obj_pos[2], target_pos[2]) - YCB_SIZE[self.object_name][2] / 2.0, 0)
            reward += 50 * lift
            condition = lift > 0.06
            if condition:  # if object off the table
                obj_target_distance = np.linalg.norm(obj_pos - target_pos)
                reward += 2.0  # bonus for lifting the object
                reward += -0.5 * np.linalg.norm(palm_pos - target_pos)  # make hand go to target
                reward += -1.5 * obj_target_distance  # make object go to target

                if obj_target_distance < 0.05:
                    reward += 1 / (max(obj_target_distance, 0.03))
                    if obj_target_distance < 0.05:
                        obj_quat = self.data.body_xquat[self.obj_bid].ravel()
                        z_axis = transforms3d.quaternions.quat2mat(obj_quat) @ np.array([0, 0, 1])
                        reward += np.arccos(z_axis[2]) * 100

                    reward += np.sum(self.check_success_particles()) * 100 / self.num_particles

        # punish water dropping
        reward -= 0.1 * dropping_ratio

        return reward

    def _setup_references(self):
        self.tank_bottom_bid = self.sim.model.body_name2id("water_tank")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id(self.object_body_name)
        self.target_sid = self.sim.model.site_name2id("pouring_site")

    def _load_model(self):
        arena = TableArena(table_full_size=(1.2, 1.2, 0.05), table_friction=(1, 0.5, 0.01), table_offset=(0, 0, 1.0),
                           bottom_pos=(0, 0, -1), has_legs=True)
        xml_file = xml_path_completion("adroit/adroit_pour.xml")
        robot = MujocoXML(xml_file)
        mesh_list = find_elements(robot.worldbody[0], tags="geom", return_first=False)
        robot_geom_names = [geom.get("name", "") for geom in mesh_list]
        self.robot_geom_names = [name for name in robot_geom_names if len(name) > 0]

        # Add YCB Object for relocation task
        object_size = YCB_SIZE[self.object_name]

        # Parse args for YCB object geom
        additional_kwargs = self.__additional_kwargs.copy()
        for key, value in additional_kwargs.items():
            if isinstance(value, (np.ndarray, tuple, list, float, int)):
                additional_kwargs[key] = array_to_string(value)
        if "condim" not in additional_kwargs:
            additional_kwargs["condim"] = "4"
        if "margin" not in additional_kwargs:
            additional_kwargs["margin"] = "0.003"

        # Add ycb object into the arena and cache its body name
        mug_init_pos = [self.mug_init_offset[0] + self.tank_pos[0],
                        self.mug_init_offset[1] + self.tank_pos[1], object_size[2] / 2]
        arena.add_ycb_object(self.object_name, pos=mug_init_pos, quat=[0.707, 0, 0, 0.707], free=True, density=1000,
                             idn=0, version="v2", **additional_kwargs)
        self.object_body_name = arena.objects[0].body_name
        object_geom = find_elements(arena.objects[0].body, "geom", return_first=False)
        self.body_geom_names = [geom.get("name", "") for geom in object_geom if
                                geom.get("name", "").startswith("collision")]
        self.object_body_name = arena.objects[0].body_name

        # Setup water tank
        mug_element = arena.objects[0].body
        site_pos = array_to_string(np.array([self.tank_pos[0] + 0.05, self.tank_pos[1], self.tank_size[2] + 0.12]))
        pouring_site = ET.Element("site", name="pouring_site", size="0.01", group="1", type="sphere", pos=site_pos,
                                  rgba="0 1 0 0")
        arena.worldbody.append(pouring_site)

        # Add water tank
        add_empty_tank(arena.worldbody, "water_tank", pos=[self.tank_pos[0], self.tank_pos[1], 0], quat=[1, 0, 0, 0],
                       size=self.tank_size, thickness=0.006, tank_color=[0.9, 0.9, 0.3, 1])
        self.num_particles = add_water_to_object(arena.worldbody, mug_element, (0.06, 0.06, 0.08),
                                                 water_center_pos=(0, 0, 0.06), ball_size=0.01)
        tank_geom = find_elements(arena.worldbody, "geom", return_first=False)
        self.tank_geom_names = [geom.get("name", "") for geom in tank_geom if "water_tank" in geom.get("name", "")]

        # Enlarge MuJoCo buffer size
        robot.size.set('nconmax', str(2e4))
        robot.size.set('njmax', str(5e3))
        robot.size.set('nstack', str(5e6))

        # Merge robot xml with table arena
        robot.merge(arena, merge_body="default")
        self.model = robot

    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        return [seed]

    @property
    def action_spec(self):
        high = np.ones_like(self.mjpy_model.actuator_ctrlrange[:, 1])
        low = -1.0 * np.ones_like(self.mjpy_model.actuator_ctrlrange[:, 0])
        return low, high

    def set_state(self, qpos, qvel):
        import mujoco_py
        assert qpos.shape == (self.mjpy_model.nq,) and qvel.shape == (self.mjpy_model.nv,)

        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def check_success_particles(self):
        # Assume that the water particles are the last body in MuJoCo model
        particle_pos = self.sim.data.body_xpos[-self.num_particles:, :].copy()
        upper_limit = (self.tank_size[:2] / 2 + self.tank_pos)
        lower_limit = (-self.tank_size[:2] / 2 + self.tank_pos)
        z = self.tank_size[2] / 2
        x_within = np.logical_and(particle_pos[:, 0] < upper_limit[0], particle_pos[:, 0] > lower_limit[0])
        y_within = np.logical_and(particle_pos[:, 1] < upper_limit[1], particle_pos[:, 1] > lower_limit[1])
        z_within = np.logical_and(particle_pos[:, 2] < z, particle_pos[:, 2] > 0)
        xy_within = np.logical_and(x_within, y_within)
        tank_within = np.logical_and(z_within, xy_within)
        return tank_within

    def check_in_mug_particles(self):
        particle_pos = self.sim.data.body_xpos[-self.num_particles:, :].copy()
        pose_mug = np.eye(4)
        pose_mug[:3, 3] = self.data.body_xpos[self.obj_bid].ravel()
        pose_mug[:3, :3] = transforms3d.quaternions.quat2mat(self.data.body_xquat[self.obj_bid].ravel())
        pose_mug_inv = inverse_pose(pose_mug)
        particle_pos_mug = particle_pos @ pose_mug_inv[:3, :3].T + pose_mug_inv[:3, 3]
        size = np.array(YCB_SIZE[self.object_name]) / 2
        # Margin z size for 0.05 since particles may out of mug
        within_mug = np.logical_and.reduce([particle_pos_mug[:, 0] < size[0], particle_pos_mug[:, 0] > -size[0],
                                            particle_pos_mug[:, 1] < size[1], particle_pos_mug[:, 1] > -size[1],
                                            particle_pos_mug[:, 2] < size[2] + 0.05, particle_pos_mug[:, 2] > -size[2]])
        return within_mug

    def check_above_particle(self, margin=0.01):
        particle_pos = self.sim.data.body_xpos[-self.num_particles:, :].copy()
        upper_limit = (self.tank_size[:2] / 2 + self.tank_pos)
        lower_limit = (-self.tank_size[:2] / 2 + self.tank_pos)
        x_within = np.logical_and(particle_pos[:, 0] < upper_limit[0] + margin,
                                  particle_pos[:, 0] > lower_limit[0] - margin)
        y_within = np.logical_and(particle_pos[:, 1] < upper_limit[1] + margin,
                                  particle_pos[:, 1] > lower_limit[1] - margin)
        z_within = particle_pos[:, 2] > 0
        xy_within = np.logical_and(x_within, y_within)
        tank_above = np.logical_and(z_within, xy_within)
        return tank_above

    @property
    def spec(self):
        this_spec = Spec(self._get_observations().shape[0], self.action_spec[0].shape[0])
        return this_spec

    def set_seed(self, seed=None):
        return self.seed(seed)

    def get_current_water_percentage(self):
        return self.check_success_particles().sum() / self.num_particles


class Spec:
    def __init__(self, observation_dim, action_dim):
        self.observation_dim = observation_dim
        self.action_dim = action_dim


def add_empty_tank(root, name, size, pos, *, thickness=0.012, quat=(1, 0, 0, 0), tank_color=(0.3, 0.3, 0.3, 1)):
    tank = create_water_tank(name, size, thickness, pos, quat, tank_color)
    root.append(tank)


def add_water_to_object(root, object_element: ET.Element, block_size, *, water_center_pos=(), ball_size=0.006,
                        water_rgba=(0.3216, 0.3569, 0.9804, 0.6)):
    if len(water_center_pos) == 0:
        water_center_pos = np.array(block_size) / 2
    spacing = str(ball_size + 0.002)
    offset = np.array(water_center_pos) + np.array(string_to_array(object_element.get("pos")))
    offset = array_to_string(offset)
    count = array_to_string((np.array(block_size) / 2 / (ball_size + 0.002)).astype(np.int))
    composite = ET.Element("composite", type="particle", spacing=spacing, offset=offset, count=count)
    geom = ET.Element("geom", size=array_to_string(ball_size), rgba=array_to_string(water_rgba))
    composite.append(geom)
    joint = ET.Element("joint", kind="main", damping="0.01")
    composite.append(joint)
    root.append(composite)
    num_counts = int(np.prod(string_to_array(count)))
    return num_counts


def create_water_tank(name, size, thickness, pos, quat, tank_color):
    if len(array_to_string(size).split(" ")) != 3:
        raise ValueError(f"Size should have length 3, but give size : {size}")
    tank = ET.Element("body", name=name, pos=array_to_string(pos), quat=array_to_string(quat))
    x, y, z = np.array(size) / 2
    thickness /= 2
    board_position = [(0, 0, thickness), (x - thickness, 0, z), (0, y - thickness, z), (-x + thickness, 0, z),
                      (0, -y + thickness, z)]
    thickness *= 4
    board_size = [(x, y, thickness), (thickness, y, z), (x, thickness, z),
                  (thickness, y, z), (x, thickness, z)]
    thickness /= 4
    visual_board_size = [(x - thickness * 2, y - thickness * 2, thickness), (thickness, y, z),
                         (x - thickness * 2, thickness, z), (thickness, y, z), (x - thickness * 2, thickness, z)]
    for i in range(5):
        geom = ET.Element("geom", type="box", pos=array_to_string(board_position[i]),
                          size=array_to_string(board_size[i]), group="2", name=f"{name}_{i}")
        visual_geom = ET.Element("geom", type="box", pos=array_to_string(board_position[i]),
                                 size=array_to_string(visual_board_size[i]), rgba=array_to_string(tank_color),
                                 contype="0", conaffinity="0", group="1")

        tank.append(visual_geom)
        tank.append(geom)
    return tank
