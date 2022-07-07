from typing import List, Dict

import mujoco_py
import numpy as np

from hand_imitation.env.environments.base import MujocoEnv
from hand_imitation.env.models.objects import YCB_SIZE
from hand_imitation.misc.data_utils import dict_has_none


class DemonstrationBase(MujocoEnv):
    def replay_state(self, demonstration: Dict[str, List]):
        observations = demonstration['observations']
        rewards = demonstration['rewards']
        sim_data = demonstration['sim_data']
        model_data = demonstration['model_data'][0]
        self.pack_mujoco_model(model_data)
        seq_len = len(observations)
        for i in range(seq_len):
            self.pack(sim_data[i])
            self.sim.forward()
            if self.has_renderer:
                for _ in range(1):
                    self.render()

    def replay_action(self, demonstration: Dict[str, List], state_freq=10):
        from mujoco_py.builder import MujocoException
        observations = demonstration['observations']
        rewards = demonstration['rewards']
        actions = demonstration['actions']
        sim_data = demonstration['sim_data']
        model_data = demonstration['model_data'][0]
        self.pack_mujoco_model(model_data)
        seq_len = len(actions)
        for i in range(seq_len):
            if i % state_freq == 0:
                self.pack(sim_data[i])
            try:
                self.step(actions[i])
            except MujocoException:
                self.sim.forward()
            if self.has_renderer:
                for _ in range(1):
                    self.render()

    def fetch_imitation_data(self, action_mean=None, action_range=None):
        obs = self._get_observations()
        reward = self.reward(None)
        action = self.compute_action(action_mean, action_range)
        return {"observations": obs, "rewards": reward, "sim_data": self.dump(), "actions": action}

    @staticmethod
    def strip(retarget_qpos_seq: List[np.ndarray], object_pose_seq: List[Dict[str, np.ndarray]]):
        while dict_has_none(object_pose_seq[0]):
            retarget_qpos_seq = retarget_qpos_seq[1:]
            object_pose_seq = object_pose_seq[1:]

        while dict_has_none(object_pose_seq[-1]):
            retarget_qpos_seq = retarget_qpos_seq[:-1]
            object_pose_seq = object_pose_seq[:-1]

        return retarget_qpos_seq, object_pose_seq

    def strip_negative_origin(self, retarget_qpos_seq: List[np.ndarray], object_pose_seq: List[Dict[str, np.ndarray]]):
        # Not only strip empty value, but also strip position where the robot arm is behind origin
        retarget_qpos_seq, object_pose_seq = self.strip(retarget_qpos_seq, object_pose_seq)
        while retarget_qpos_seq[0][2] < 0:
            retarget_qpos_seq = retarget_qpos_seq[1:]
            object_pose_seq = object_pose_seq[1:]
        return retarget_qpos_seq, object_pose_seq

    def hindsight_replay_sequence(self, retarget_qpos_seq: List[np.ndarray],
                                  object_pose_seq: List[Dict[str, np.ndarray]], reference_object_name: str,
                                  init_object_lift=None):
        joint2world = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
        init_hand_pos = joint2world @ retarget_qpos_seq[0][:3].copy()
        if init_object_lift is None:
            init_object_lift = object_pose_seq[0][reference_object_name][2, 3] - YCB_SIZE[reference_object_name][2]
        init_offset_position = np.array([init_hand_pos[0], init_hand_pos[1], init_object_lift])
        init_hand_joint_offset = joint2world.T @ init_offset_position
        # init_hand_offset_rotation = init_hand_rotation.T
        # init_hand_rotation = transforms3d.euler.euler2mat(*retarget_qpos_seq[0][3:6])

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

    def get_object_joint_qpos_indices(self, object_name: str, must_unique=False):
        all_joint_names = self.mjpy_model.joint_names
        object_joint_names = []
        for joint_name in all_joint_names:
            if object_name in joint_name:
                object_joint_names.append(joint_name)
        if must_unique and len(object_joint_names) > 1:
            raise RuntimeError(f"Joint names of object {object_name} is not unique, found {len(object_joint_names)}")

        return self.get_joint_qpos_indices_from_names(object_joint_names)

    def compute_actuator_input(self):
        gain = self.mjpy_model.actuator_gainprm[:, 0]
        bias = self.mjpy_model.actuator_biasprm[:, 1]
        acc = gain.shape[0]
        mujoco_py.functions.mj_inverse(self.mjpy_model, self.data)
        qfrc = self.data.qfrc_inverse[:acc] - self.data.qfrc_bias[:acc] + self.data.qfrc_constraint[:acc]
        qpos = self.data.qpos[:gain.shape[0]]
        actuation = (qfrc - bias * qpos) / gain
        return actuation

    def compute_action(self, action_mean, action_span):
        action = self.compute_actuator_input()
        if action_mean is not None and action_span is not None:
            action = (action - action_mean) / action_span
            action = np.clip(action, -1, 1)
        return action


class LPFilter:
    def __init__(self, video_freq, cutoff_freq):
        dt = 1 / video_freq
        wc = cutoff_freq * 2 * np.pi
        y_cos = 1 - np.cos(wc * dt)
        self.alpha = -y_cos + np.sqrt(y_cos ** 2 + 2 * y_cos)
        self.y = 0

    def next(self, x):
        self.y = self.y + self.alpha * (x - self.y)
        return self.y

    def init_value(self, y):
        self.y = y
