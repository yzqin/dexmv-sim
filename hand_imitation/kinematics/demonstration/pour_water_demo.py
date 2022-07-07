import time
from typing import List, Dict

import numpy as np
import transforms3d

from hand_imitation.env.environments.mug_pour_water_env import WaterPouringEnv
from hand_imitation.kinematics.demonstration.base import DemonstrationBase, LPFilter
from hand_imitation.misc.data_utils import min_jerk_interpolate_replay_sequence


class WaterPouringDemonstration(DemonstrationBase, WaterPouringEnv):
    def __init__(self, has_renderer, **kwargs):
        super().__init__(has_renderer, **kwargs)
        self.filter = LPFilter(30, 5)
        self.init_sim_data = self.dump()
        self.init_model_data = self.dump_mujoco_model()
        object_qpos_indices = self.get_object_joint_qpos_indices("mug")
        self.object_trans_qpos_indices = object_qpos_indices[:3]
        self.object_rot_qpos_indices = object_qpos_indices[3:]

    def play_hand_object_seq(self, retarget_qpos_seq: List[np.ndarray], object_pose_seq: List[Dict[str, np.ndarray]],
                             name="undefined"):
        tic = time.time()

        # Processing Sequence Data
        reference_object_name = self.object_name
        action_time_step = self.control_timestep
        collect_time_step = 1 / 25.0
        retarget_qpos_seq, object_pose_seq = self.strip_negative_origin(retarget_qpos_seq, object_pose_seq)
        retarget_qpos_seq, object_pose_seq = self.hindsight_replay_sequence(retarget_qpos_seq, object_pose_seq,
                                                                            reference_object_name)
        self.hind_sight_environment_model(retarget_qpos_seq, object_pose_seq, reference_object_name)

        retarget_qpos_seq, retarget_qvel_seq, retarget_qacc_seq, object_pose_seq = min_jerk_interpolate_replay_sequence(
            retarget_qpos_seq, object_pose_seq, action_time_step, collect_time_step)

        result = {}
        imitation_data = []
        num_samples = len(retarget_qpos_seq)
        result["model_data"] = [self.dump_mujoco_model()]

        for i in range(num_samples):
            object_pose = object_pose_seq[i][self.object_name]
            qpos = self.filter.next(retarget_qpos_seq[i][:])
            self.sim.data.qpos[:qpos.shape[0]] = qpos
            self.sim.data.qvel[:qpos.shape[0]] = np.clip(retarget_qvel_seq[i][:], -3, 3)
            self.sim.data.qacc[:qpos.shape[0]] = np.clip(retarget_qacc_seq[i][:], -5, 5)
            self.sim.data.qpos[self.object_trans_qpos_indices] = object_pose[:3, 3]
            self.sim.data.qpos[self.object_rot_qpos_indices] = transforms3d.quaternions.mat2quat(object_pose[:3, :3])
            imitation_data.append(self.fetch_imitation_data(self.act_mid, self.act_rng))

            if self.has_renderer:
                for _ in range(1):
                    self.render()
            self.sim.forward()

        # Pack entry together by key
        step_size = len(imitation_data)
        for key in ['observations', 'rewards', 'actions']:
            result[key] = np.stack([imitation_data[i][key] for i in range(step_size)], axis=0)
        result['sim_data'] = [imitation_data[i]["sim_data"] for i in range(step_size)]

        # Verbose
        duration = time.time() - tic
        print(f"Generating demo data {name} with {num_samples} samples takes {duration} seconds.")
        self.pack(self.init_sim_data)
        self.pack_mujoco_model(self.init_model_data)
        return result

    def hind_sight_environment_model(self, retarget_qpos_seq: List[np.ndarray],
                                     object_pose_seq: List[Dict[str, np.ndarray]], reference_object_name: str):
        retarget_qpos_seq, object_pose_seq = self.hindsight_replay_sequence(retarget_qpos_seq, object_pose_seq, "mug")
        joint2world = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
        init_offset = np.array([-0.2, 0, 0])
        init_hand_joint_offset = joint2world.T @ init_offset

        for i in range(len(retarget_qpos_seq)):
            retarget_qpos_seq[i][:3] = retarget_qpos_seq[i][:3] - init_hand_joint_offset
        for i in range(len(object_pose_seq)):
            if "mug" in object_pose_seq[i]:
                object_pose_seq[i]["mug"][:3, 3] -= init_offset

        for object_name, pose in object_pose_seq[-1].items():
            if object_name == reference_object_name:
                target_body_pos = pose[:3, 3]
                target_body_quat = transforms3d.quaternions.mat2quat(pose[:3, :3])
                self.mjpy_model.site_pos[self.mjpy_model.site_name2id("pouring_site"), 0:3] = target_body_pos
                self.mjpy_model.body_pos[self.mjpy_model.body_name2id("water_tank"), 0:3] = target_body_pos + np.array(
                    [-0.05, 0, -0.08])
                self.mjpy_model.body_pos[self.mjpy_model.body_name2id("water_tank"), 2] = self.tank_size[2] / 2
        # site_pos = array_to_string(np.array([self.tank_pos[0] + 0.05, self.tank_pos[1], self.tank_size[2] + 0.08]))

        mug_init_pos = object_pose_seq[0][reference_object_name][:3, 3]
        mug_body_index = self.mjpy_model.body_name2id(self.object_body_name)
        mug_offset = mug_init_pos - self.mjpy_model.body_pos[mug_body_index, :]
        particle_pos = np.ones_like(self.original_particle_pos) * -10
        self.sim.data.qpos[-self.num_particles * 3:] = particle_pos.ravel()
