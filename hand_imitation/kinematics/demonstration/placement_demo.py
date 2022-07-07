import time
import warnings
from typing import List, Dict

import numpy as np
import transforms3d

from hand_imitation.env.environments.mug_place_object_env import MugPlaceObjectEnv
from hand_imitation.env.models.objects import YCB_SIZE
from hand_imitation.kinematics.demonstration.base import DemonstrationBase, LPFilter
from hand_imitation.misc.data_utils import min_jerk_interpolate_replay_sequence
from hand_imitation.misc.joint_utils import filter_position_sequence


class PlacementDemonstration(DemonstrationBase, MugPlaceObjectEnv):
    def __init__(self, has_renderer, object_scale=1.0, **kwargs):
        super().__init__(has_renderer, -1, object_scale=object_scale, **kwargs)
        self.filter = LPFilter(30, 5)
        self.init_sim_data = self.dump()
        self.init_model_data = self.dump_mujoco_model()
        object_qpos_indices = self.get_object_joint_qpos_indices(self.object_name)
        self.object_trans_qpos_indices = object_qpos_indices[:3]
        self.object_rot_qpos_indices = object_qpos_indices[3:]

    def play_hand_object_seq(self, retarget_qpos_seq: List[np.ndarray], object_pose_seq: List[Dict[str, np.ndarray]],
                             name="undefined"):
        tic = time.time()

        # Processing Sequence Data
        reference_object_name = self.object_name
        action_time_step = self.control_timestep
        collect_time_step = 1 / 25.0
        retarget_qpos_seq, object_pose_seq = self.strip_specific_objects(retarget_qpos_seq, object_pose_seq)
        if object_pose_seq[-1][self.object_name][2, 3] < 0:
            warnings.warn("Target position has a z < 0, skip it!")
            return None

        # Filter Object Sequence
        temp_object_pos_seq = []
        temp_object_rot_seq = []
        for k in range(len(object_pose_seq)):
            if self.object_name in object_pose_seq[k]:
                temp_object_pos_seq.append(object_pose_seq[k][self.object_name][:3, 3])
                axis, angle = transforms3d.quaternions.quat2axangle(
                    transforms3d.quaternions.mat2quat(object_pose_seq[k][self.object_name][:3, :3]))
                temp_object_rot_seq.append(axis * angle)
        object_position_filter = filter_position_sequence(np.array(temp_object_pos_seq), wn=5, fs=100)
        object_lie_filter = filter_position_sequence(np.array(temp_object_rot_seq), wn=5, fs=200)
        num = 0
        for k in range(len(object_pose_seq)):
            if "banana" in object_pose_seq[k]:
                angle = np.linalg.norm(object_lie_filter[num])
                axis = object_lie_filter[num] / (angle + 1e-6)
                rotation = transforms3d.axangles.axangle2mat(axis, angle)
                object_pose_seq[k][self.object_name][:3, :3] = rotation
                object_pose_seq[k][self.object_name][:3, 3] = object_position_filter[num]
                num += 1

        retarget_qpos_seq, object_pose_seq = self.hindsight_replay_sequence(retarget_qpos_seq, object_pose_seq,
                                                                            reference_object_name)

        retarget_qpos_seq, retarget_qvel_seq, retarget_qacc_seq, object_pose_seq = min_jerk_interpolate_replay_sequence(
            retarget_qpos_seq, object_pose_seq, action_time_step, collect_time_step)

        result = {}
        imitation_data = []
        num_samples = len(retarget_qpos_seq)
        result["model_data"] = [self.dump_mujoco_model()]
        finish_count = 0

        for i in range(num_samples):
            if i < 30:
                continue
            object_pose = object_pose_seq[i][self.object_name]
            qpos = self.filter.next(retarget_qpos_seq[i][:])
            self.sim.data.qpos[:qpos.shape[0]] = qpos
            self.sim.data.qvel[:qpos.shape[0]] = np.clip(retarget_qvel_seq[i][:], -3, 3)
            self.sim.data.qacc[:qpos.shape[0]] = np.clip(retarget_qacc_seq[i][:], -10, 10)
            self.sim.data.qpos[self.object_trans_qpos_indices] = object_pose[:3, 3]
            self.sim.data.qpos[self.object_rot_qpos_indices] = transforms3d.quaternions.mat2quat(object_pose[:3, :3])

            imitation_data.append(self.fetch_imitation_data(self.act_mid, self.act_rng))

            # break loop if already finished
            if self.reward(None) > 15:
                finish_count += 1
            else:
                finish_count = 0
            if finish_count >= 5:
                break

            if self.has_renderer:
                for _ in range(1):
                    self.render()
            self.sim.forward()

        if self.reward(None) < 15:
            warnings.warn("Demonstration not success, skip it!")
            # return None

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

    def hindsight_replay_sequence(self, retarget_qpos_seq: List[np.ndarray],
                                  object_pose_seq: List[Dict[str, np.ndarray]], reference_object_name: str,
                                  init_object_lift=None):
        # mug_pose = np.array([[-0.1696, 0.7044, 0.6892, -0.4556],
        #                      [0.9814, 0.1844, 0.053, 0.096],
        #                      [-0.0897, 0.6854, -0.7226, 0.8026],
        #                      [0., 0., 0., 1.]]) @ np.array([[-0.66328378, 0.74287763, 0.10231653, 0.03552204],
        #                                                     [0.64820283, 0.50248807, 0.57180128, -0.04642461],
        #                                                     [0.36730237, 0.4493042, -0.81330573, 0.93256695],
        #                                                     [0., 0., 0., 1.]])
        mug_pose = object_pose_seq[0]["mug"]
        joint2world = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
        final_object_pos = mug_pose[:3, 3]
        if init_object_lift is None:
            init_object_lift = final_object_pos[2] - YCB_SIZE["mug"][2]
        init_offset_position = self.compute_container_pos(object_pose_seq[-1][reference_object_name], init_object_lift)
        init_hand_joint_offset = joint2world.T @ init_offset_position

        if len(retarget_qpos_seq) != len(object_pose_seq):
            raise RuntimeError(f"Length of robot retargeting result does not match with object pose result")
        for i in range(len(retarget_qpos_seq)):
            current_qpos = retarget_qpos_seq[i]
            current_qpos[:3] = current_qpos[:3] - init_hand_joint_offset
            retarget_qpos_seq[i] = current_qpos

            for object_name in object_pose_seq[0].keys():
                # Duplicate previous data for empty pose
                if object_name not in object_pose_seq[i] or object_pose_seq[i][object_name] is None:
                    object_pose_seq[i][object_name] = object_pose_seq[i - 1][object_name]
                # Hindsight object pose data for non-empty pose
                else:
                    pose = object_pose_seq[i][object_name]
                    current_object_pose = pose
                    current_object_pose[:3, 3] = current_object_pose[:3, 3] - init_offset_position
                    object_pose_seq[i][object_name] = current_object_pose

        return retarget_qpos_seq, object_pose_seq

    @staticmethod
    def compute_container_pos(banana_mat, init_container_lift):
        y_axis = banana_mat[:3, 1]
        lean_angle = np.arccos(np.sum(y_axis * np.array([0, 0, 1])))
        upside_down = lean_angle > np.deg2rad(60)
        banana_pos = banana_mat[:3, 3]
        if upside_down:
            edge_local_pos = np.array([0.073, 0.072, -0.011])  # magical number from banana mesh
            edge_pos = banana_mat[:3, :3] @ edge_local_pos + banana_pos
            center_pos = (edge_pos * 1 + banana_pos * 1) / 2
        else:
            edge_local_pos = np.array([-0.010, -0.101, -0.003])  # magical number from banana mesh
            edge_pos = banana_mat[:3, :3] @ edge_local_pos + banana_pos
            center_pos = (edge_pos * 1 + banana_pos * 2) / 3

        init_offset_z = min(init_container_lift, center_pos[2] - 0.05)
        # init_offset_z = init_container_lift
        init_offset_position = np.array([center_pos[0], center_pos[1], init_offset_z])
        return init_offset_position

    def strip_specific_objects(self, retarget_qpos_seq: List[np.ndarray], object_pose_seq: List[Dict[str, np.ndarray]]):
        # Not only strip empty value, but also strip position where the robot arm is behind origin
        retarget_qpos_seq, object_pose_seq = self.strip_negative_origin(retarget_qpos_seq, object_pose_seq)
        while "banana" not in object_pose_seq[0] or "mug" not in object_pose_seq[0]:
            retarget_qpos_seq = retarget_qpos_seq[1:]
            object_pose_seq = object_pose_seq[1:]
        return retarget_qpos_seq, object_pose_seq
