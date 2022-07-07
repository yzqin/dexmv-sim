import numpy as np
import transforms3d
import trimesh

from hand_imitation.env.environments.base import MujocoEnv
from hand_imitation.env.models import TableArena
from hand_imitation.env.models.base import MujocoXML
from hand_imitation.env.models.objects import YCB_SIZE
from hand_imitation.env.utils.mjcf_utils import xml_path_completion, array_to_string, find_elements
from hand_imitation.env.utils.random import np_random


class MugPlaceObjectEnv(MujocoEnv):
    def __init__(self, has_renderer, render_gpu_device_id=-1, object_scale=1, randomness_scale=1, mug_scale=2,
                 large_force=True, load_eval_mesh=False, **kwargs):
        self.np_random = None
        self.seed()
        self.__additional_kwargs = kwargs
        self.object_name = "banana"
        self.randomness_scale = randomness_scale
        self.object_scale = object_scale
        self.mug_scale = mug_scale
        self.large_force = large_force
        self.load_eval_mesh = load_eval_mesh

        super().__init__(has_renderer=has_renderer, has_offscreen_renderer=False, render_camera=None,
                         render_gpu_device_id=render_gpu_device_id, control_freq=100, horizon=200, ignore_done=True,
                         hard_reset=False)

        # Setup action range
        self.act_mid = np.mean(self.mjpy_model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.mjpy_model.actuator_ctrlrange[:, 1] - self.mjpy_model.actuator_ctrlrange[:, 0])
        if load_eval_mesh:
            mesh_name = f"{self.object_name}_visual_mesh"
            msh_file = find_elements(self.model.asset, "mesh", {"name": mesh_name}).get("file")
            obj_file = msh_file.replace("_blender.msh", ".obj")
            self.object_mesh = trimesh.load(obj_file)
            self.object_volume = self.object_mesh.intersection([self.object_mesh]).volume  # reduce numerical error
            mug_size = np.array([0.045, 0.08]) * mug_scale
            self.mug_mesh = trimesh.primitives.Cylinder(radius=mug_size[0], height=mug_size[1])
            self.mug_mesh.apply_translation(self.data.body_xpos[self.mug_bid])

    def _pre_action(self, action, policy_step=False):
        action = np.clip(action, -1.0, 1.0)
        action = self.act_mid + action * self.act_rng  # mean center and scale
        self.sim.data.ctrl[:] = action

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.forward()
        self.sim.set_state(self.sim_state_initial)
        self.sim.forward()
        self.sim.data.qpos[30] = self.np_random.uniform(low=-0.15, high=0.15) * self.randomness_scale
        self.sim.data.qpos[31] = self.np_random.uniform(low=0.0, high=0.1) * self.randomness_scale + 0.1

    def _get_observations(self):
        qp = self.data.qpos.ravel()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        obj_quat = self.data.body_xquat[self.obj_bid].ravel()
        mug_pos = self.data.body_xpos[self.mug_bid].ravel()
        return np.concatenate([qp[:30], palm_pos - obj_pos, palm_pos - mug_pos, obj_pos - mug_pos, obj_quat])

    def reward(self, action):
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        obj_quat = self.data.body_xquat[self.obj_bid].ravel()
        obj_rot = transforms3d.quaternions.quat2mat(obj_quat)
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        mug_top_pos = self.data.body_xpos[self.mug_bid].ravel() + [0, 0, YCB_SIZE["mug"][2] / 2 * self.mug_scale]
        is_contact = self.check_contact(self.body_geom_names, self.robot_geom_names)
        max_lift_height = mug_top_pos[2] + YCB_SIZE["banana"][1] * self.object_scale / 3

        reward = -0.1 * np.linalg.norm(palm_pos - obj_pos)  # take hand to object
        obj_target_distance_xy = np.linalg.norm(obj_pos[:2] - mug_top_pos[:2])
        vertical_angle = np.arccos(obj_rot[2, 1])
        lift = max(min(obj_pos[2], max_lift_height) - YCB_SIZE[self.object_name][2] / 2.0, 0)
        if is_contact:
            reward += 0.2
            reward += lift * 2
            if lift > mug_top_pos[2]:  # if object off the table
                reward += 0.2
                reward += -0.5 * np.linalg.norm(palm_pos[:2] - mug_top_pos[:2])
                reward += -1 * obj_target_distance_xy

                # 0 vertical angle means totally vertical
                if obj_target_distance_xy < 0.05:
                    reward += 0.5
                    reward += (np.pi / 2 - vertical_angle) * 0.5
                if obj_target_distance_xy < 0.03:
                    reward += 0.3

        if obj_target_distance_xy < 0.03 and vertical_angle < np.deg2rad(45):
            reward += (max_lift_height - lift) * 12
            if lift < (mug_top_pos[2] + max_lift_height) / 2:
                hand_joints_mean = np.abs(self.data.qpos[6:30].copy()).mean()
                reward += 0.5 + (np.pi / 2 - hand_joints_mean) * 2

            # Compensate reward
            if not is_contact or lift <= mug_top_pos[2]:
                reward += 1 + -1 * obj_target_distance_xy + 0.5 + (np.pi / 2 - vertical_angle) * 0.5
            if not is_contact:
                reward += 1.2 + 2 * lift

        return reward

    def _setup_references(self):
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id(self.object_body_name)
        self.mug_bid = self.sim.model.body_name2id("mug_0")

    def _load_model(self):
        arena = TableArena(table_full_size=(1.2, 1.2, 0.05), table_friction=(1, 0.5, 0.01), table_offset=(0, 0, 1.0),
                           bottom_pos=(0, 0, -1), has_legs=True)
        xml_file = xml_path_completion("adroit/adroit_placement.xml")
        robot = MujocoXML(xml_file)
        mesh_list = find_elements(robot.worldbody[0], tags="geom", return_first=False)
        robot_geom_names = [geom.get("name", "") for geom in mesh_list]
        self.robot_geom_names = [name for name in robot_geom_names if len(name) > 0]

        # Parse args for YCB object geom
        additional_kwargs = self.__additional_kwargs.copy()
        for key, value in additional_kwargs.items():
            if isinstance(value, (np.ndarray, tuple, list, float, int)):
                additional_kwargs[key] = array_to_string(value)
        if "condim" not in additional_kwargs:
            additional_kwargs["condim"] = "4"

        # Add YCB Object for placement task
        object_size = YCB_SIZE[self.object_name]
        arena.add_ycb_object(self.object_name, pos=[0, 0.1, object_size[2] / 2 * self.object_scale],
                             quat=[0.707, 0, 0, 0.707], free=True, density=1000, idn=0, scale=self.object_scale,
                             **additional_kwargs)

        # Add mug as the target object for placement task
        mug_size = YCB_SIZE["mug"]
        arena.add_ycb_object("mug", pos=[0, 0, mug_size[2] / 2 * self.mug_scale], quat=[1, 0, 0, 0], free=False,
                             version="xml", scale=self.mug_scale, margin="0.03", gap="0.03")

        # Add ycb object into the arena and cache its body name
        self.object_body_name = arena.objects[0].body_name
        object_geom = find_elements(arena.objects[0].body, "geom", return_first=False)
        self.body_geom_names = [geom.get("name", "") for geom in object_geom if
                                geom.get("name", "").startswith("collision")]
        self.object_body_name = arena.objects[0].body_name

        # Reduce actuator force if not large force
        if not self.large_force:
            for actuator in find_elements(robot.actuator, "general", return_first=False):
                if "A_ARR" in actuator.get("name"):
                    actuator.set("gainprm", "250 0 0 0 0 0 0 0 0 0")
            print(f"Modify actuator force from 500 gain to 250 gain since large_force is set to False")

        # Merge robot xml with table arena
        robot.merge(arena, merge_body="default")
        self.model = robot
        self.model.save_model("placement_temp.xml")

    def compute_intersection_rate(self):
        if not self.load_eval_mesh:
            raise RuntimeError(f"Eval mesh is not loaded. Please set the load_eval_mesh=True in constructor")
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        obj_quat = self.data.body_xquat[self.obj_bid].ravel()
        object_mat = np.eye(4)
        object_mat[:3, :3] = transforms3d.quaternions.quat2mat(obj_quat)
        object_mat[:3, 3] = obj_pos
        self.object_mesh.apply_transform(object_mat)
        intersect = self.mug_mesh.intersection([self.object_mesh])
        volume = 0 if isinstance(intersect, trimesh.Scene) else intersect.volume
        percentage = volume / self.object_volume

        self.object_mesh.apply_transform(np.linalg.inv(object_mat))
        return percentage

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

    @property
    def spec(self):
        this_spec = Spec(self._get_observations().shape[0], self.action_spec[0].shape[0])
        return this_spec

    def set_seed(self, seed=None):
        return self.seed(seed)


class Spec:
    def __init__(self, observation_dim, action_dim):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
