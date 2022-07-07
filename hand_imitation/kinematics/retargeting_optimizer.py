import time
import warnings
from typing import List

import numpy as np

from hand_imitation.env.models.base import MujocoXML
from hand_imitation.env.utils.errors import XMLError
from hand_imitation.env.utils.mjcf_utils import find_elements, find_parent, string_to_array
from hand_imitation.kinematics.optimizer import PositionOptimizer, KinematicChain
from hand_imitation.misc.joint_utils import get_robot_joint_pos_from_hand_frame, filter_position_sequence


class NaiveOptimizationRetargeting:
    def __init__(self, xml_filename, target_bodies=(), has_joint_limits=True):
        self.xml = MujocoXML(xml_filename)
        self._clean_xml()
        self.xml.save_model("temp.xml")

        self.optimizer = PositionOptimizer(self.xml.get_xml(), body_names=list(target_bodies))
        self.has_joint_limits = has_joint_limits
        if has_joint_limits:
            self.joint_limits = self._get_joint_limit()
            self.optimizer.set_joint_limit(self.joint_limits)

    def _clean_xml(self):
        x = self.xml
        for entry in [x.asset, x.sensor, x.tendon, x.equality, x.contact, x.actuator]:
            x.root.remove(entry)

        for tag in ["geom", "site"]:
            elements = find_elements(x.worldbody, tag, return_first=False)
            if elements:
                for element in elements:
                    parent = find_parent(x.worldbody, element)
                    parent.remove(element)

    def _parse_xml_joint_limit(self):
        joints = find_elements(self.xml.worldbody, tags="joint", return_first=False)
        joint_limits = {}
        for joint in joints:
            limit = string_to_array(joint.get("range"))
            name = joint.get("name") or "undefined"
            joint_limits[name] = limit
        return joint_limits

    def _get_joint_limit(self):
        chain_joints = self.optimizer.chain.get_variable_joints()
        chain_links = self.optimizer.chain.get_variable_links()
        xml_joint_limits = self._parse_xml_joint_limit()
        joint_limits = np.zeros([len(chain_joints), 2])
        for i, joint in enumerate(chain_joints):
            if not joint.name:
                raise XMLError(f"For retargeting purpose, joint under body {chain_links[i].body.name} must have a name")
            limit = xml_joint_limits[joint.name]
            joint_limits[i, :] = limit
        return joint_limits

    def retarget(self, pos_sequence: List[np.ndarray], name="", init_qpos=None, verbose=True):
        num_sample = len(pos_sequence)
        if num_sample == 0:
            warnings.warn(f"Sequence {name} has no data. Will skip it...")
        expected_shape = (len(self.optimizer.body_names), 3)
        if pos_sequence[0].shape != expected_shape:
            raise ValueError(f"Expect each sample in joint sequence to have shape {expected_shape}. "
                             f"But get shape {pos_sequence[0].shape} for sequence {name}")

        # Parse init qpos
        if init_qpos is None and self.has_joint_limits:
            init_qpos = self.joint_limits.mean(axis=1)

        # Retargeting
        tic = time.time()
        last_qpos = init_qpos
        retarget_qpos = []
        for i in range(num_sample):
            last_qpos = self.optimizer.retarget(pos_sequence[i], last_qpos, verbose)
            retarget_qpos.append(last_qpos)
        duration = time.time() - tic
        print(f"{self.__class__.__name__} sequence {name} with {num_sample} samples takes {duration} seconds.")
        return retarget_qpos


class ChainMatchingPositionKinematicsRetargeting:
    def __init__(self, xml_filename, target_bodies, has_joint_limits=True, has_global_pose_limits=True):
        self.xml = MujocoXML(xml_filename)
        self._clean_xml()
        self.chain = KinematicChain.build_from_mjcf(self.xml.get_xml())

        self.optimizer = PositionOptimizer(self.xml.get_xml(), body_names=list(target_bodies))

        joints = self.chain.get_variable_joints()
        root_joints = [joint.name for joint in joints if joint.name.startswith(("AR", "WR"))]
        self.matching_joints = [joint.name for joint in joints if joint.name not in root_joints]

        # Set joint limit, note that the first 6 joint is global joint which controls the root pose of hand
        self.has_joint_limits = has_joint_limits
        joint_limits = np.ones_like(self._get_joint_limit())
        joint_limits[:, 0] = -1e4  # a large value is equivalent to no limit
        joint_limits[:, 1] = 1e4
        if has_joint_limits:
            # Modify the joint limit for wrist joint to avoid large bias
            joint_limits[6:] = self._get_joint_limit()[6:]
            joint_limits[6] = [-0.175, 0.175]
            joint_limits[6] = [-0.436, 0.436]
        if has_global_pose_limits:
            joint_limits[:6] = self._get_joint_limit()[:6]
        if has_joint_limits or has_global_pose_limits:
            self.optimizer.set_joint_limit(joint_limits)
        self.joint_limits = joint_limits

    def _clean_xml(self):
        x = self.xml
        for entry in [x.asset, x.sensor, x.tendon, x.equality, x.contact, x.actuator]:
            x.root.remove(entry)

        for tag in ["geom", "site"]:
            elements = find_elements(x.worldbody, tag, return_first=False)
            if elements:
                for element in elements:
                    parent = find_parent(x.worldbody, element)
                    parent.remove(element)

    def _parse_xml_joint_limit(self):
        joints = find_elements(self.xml.worldbody, tags="joint", return_first=False)
        joint_limits = {}
        for joint in joints:
            limit = string_to_array(joint.get("range"))
            name = joint.get("name") or "undefined"
            joint_limits[name] = limit
        return joint_limits

    def _get_joint_limit(self):
        xml_joint_limits = self._parse_xml_joint_limit()
        joint_limits = np.zeros([len(self.chain.get_variable_joints()), 2])
        for i, joint in enumerate(self.chain.get_variable_joints()):
            if not joint:
                raise XMLError(f"For retargeting purpose, joint under body {joint} must have a name")
            limit = xml_joint_limits[joint.name]
            joint_limits[i, :] = limit
        return joint_limits

    def retarget(self, pos_sequence: List[np.ndarray], frame_sequence: List[np.ndarray], name="", init_qpos=None,
                 verbose=True):
        num_sample = len(frame_sequence)
        if num_sample == 0:
            warnings.warn(f"Sequence {name} has no data. Will skip it...")
        expected_shape = (16, 4, 4)
        if frame_sequence[0].shape != expected_shape:
            raise ValueError(f"Expect each sample in joint sequence to have shape {expected_shape}. "
                             f"But get shape {frame_sequence[0].shape} for sequence {name}")

        # Parse init qpos
        if init_qpos is None and self.has_joint_limits:
            init_qpos = self.joint_limits.mean(axis=1)
        elif init_qpos is None:
            init_qpos = np.zeros(self.optimizer.dof)

        # Setup retargeting joints
        all_joint_names = [joint.name for joint in self.chain.get_variable_joints()]
        robot_hand_optimization_joint_indices = np.arange(self.optimizer.dof)
        matching_joint_index = np.array([i for i in range(self.optimizer.dof) if
                                         all_joint_names[i] in self.matching_joints], dtype=int)
        tic = time.time()

        # Filtering joint sequence
        pos_sequence = filter_position_sequence(np.array(pos_sequence), wn=5, fs=100)
        last_qpos = init_qpos
        retarget_qpos = []
        for i in range(num_sample):
            # Get joint matching result and cutoff to joint limit
            kinematics_matching_joint = get_robot_joint_pos_from_hand_frame(frame_sequence[i], self.matching_joints)
            if self.has_joint_limits:
                matching_joint_limits = self.joint_limits[matching_joint_index]
                kinematics_matching_joint = np.clip(kinematics_matching_joint, matching_joint_limits[:, 0],
                                                    matching_joint_limits[:, 1])

            # Using the joint matching result as initial estimation of qpos
            last_qpos[matching_joint_index] = kinematics_matching_joint

            # Optimize global joint and LF
            optimization_joint = np.array(
                self.optimizer.retarget(pos_sequence[i], last_qpos, verbose))[robot_hand_optimization_joint_indices]

            # Merge joint matching result with optimization result
            last_qpos = np.zeros(self.optimizer.dof)
            last_qpos[matching_joint_index] = kinematics_matching_joint
            last_qpos[robot_hand_optimization_joint_indices] = optimization_joint
            retarget_qpos.append(last_qpos)
        duration = time.time() - tic
        print(f"{self.__class__.__name__} sequence {name} with {num_sample} samples takes {duration} seconds.")
        return retarget_qpos
