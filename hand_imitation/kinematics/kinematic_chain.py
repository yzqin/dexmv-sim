from typing import List, Optional

import numpy as np
import transforms3d

from hand_imitation.kinematics.mjcf_parser import from_xml_string
from hand_imitation.misc.pose_utils import skew_matrix


class Body:
    def __init__(self, name="", pose=np.eye(4)):
        self.name = name
        self.pose = pose


class Joint:
    TYPES = ['revolute', 'slide']

    def __init__(self, name, joint_type='fixed', axis=np.array([0.0, 0.0, 1.0])):
        self.name = name
        self.type = joint_type
        self.axis = axis / np.linalg.norm(axis)
        if joint_type not in Joint.TYPES:
            raise TypeError(f"Joint type {joint_type} is invalid")

    def transform(self, variable):
        delta_pose = np.eye(4)
        if self.type == 'revolute':
            delta_pose[:3, :3] = transforms3d.axangles.axangle2mat(self.axis, variable, is_normalized=True)
        elif self.type == 'slide':
            delta_pose[:3, 3] = self.axis * variable
        return delta_pose

    def __repr__(self):
        return f"Joint {self.name} with type {self.type}"


class KinematicChain:
    def __init__(self, body, joints=(), children=()):
        self.body = body
        self.joints: List[Joint] = list(joints)
        self.children: List[KinematicChain] = list(children)
        self.parent: Optional[KinematicChain] = None

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    @property
    def dof(self):
        return len(self.joints)

    @property
    def chain_dof(self):
        dof = self.dof
        for child in self.children:
            dof += child.chain_dof
        return dof

    def __str__(self, level=0):
        string = " \t" * level + f"{self.body.name}: {self.dof} dof \n"
        for child in self.children:
            string += child.__str__(level + 1)
        return string

    def __repr__(self):
        return f"Link with body {self.body.name}"

    def get_variable_joints(self):
        joints = []
        for joint in self.joints:
            joints.append(joint)
        for child in self.children:
            joints += child.get_variable_joints()
        return joints

    def get_variable_links(self):
        links = []
        for joint in self.joints:
            links.append(self)
        for child in self.children:
            links += child.get_variable_links()
        return links

    def get_link_dict(self):
        bodies = {self.body.name: self}
        for child in self.children:
            bodies.update(child.get_link_dict())
        return bodies

    def transform(self, variables):
        # Compute the transformation from parent body to current body, frame construction follow mujoco convention
        if len(variables) != len(self.joints):
            raise RuntimeError(
                f"Articulation link with body {self.body.name} have {len(self.joints)} joints. "
                f"But {len(variables)} variables were given")

        linked_transformation = self.body.pose
        for i, variable in enumerate(variables):
            delta_pose = self.joints[i].transform(variable)
            linked_transformation = linked_transformation @ delta_pose

        return linked_transformation

    def forward_kinematics(self, variables, parent_pose=np.eye(4)):
        current_pose = parent_pose @ self.transform(variables[: self.dof])
        poses = {self.body.name: current_pose}
        index = self.dof
        for child in self.children:
            child_dof = child.chain_dof
            child_variable = variables[index: index + child_dof]
            index += child_dof
            child_pose_dict = child.forward_kinematics(child_variable, current_pose)
            poses.update(child_pose_dict)
        return poses

    def jacobian(self, variables, body_names: List[str], link_poses: dict = None, position_only=True):
        # assert position_only
        link_dict = self.get_link_dict()
        joints = self.get_variable_joints()
        links = [link_dict[name] for name in body_names]
        jacobians = []
        if not link_poses:
            link_poses = self.forward_kinematics(variables)
        dof = self.chain_dof
        for link in links:
            current_link = link
            jacobian = np.zeros([3, dof]) if position_only else np.zeros([6, dof])
            target_pos = link_poses[link.body.name][:3, 3]
            while current_link:
                if current_link.parent:
                    pose_current2root = link_poses[current_link.parent.body.name] @ current_link.body.pose
                else:
                    pose_current2root = current_link.body.pose
                joint_trans_cache = [None] * current_link.dof
                for i, joint in enumerate(current_link.joints):
                    joint_trans_cache[i] = joint.transform(variables[joints.index(joint)])
                    pose_current2root = pose_current2root @ joint_trans_cache[i]
                    omega = pose_current2root[:3, :3] @ joint.axis
                    if joint.type == "revolute":
                        delta_pos = target_pos - pose_current2root[:3, 3]
                        linear_velocity = np.cross(omega, delta_pos)
                        angular_velocity = omega
                    else:
                        linear_velocity = omega
                        angular_velocity = np.zeros(3)
                    jacobian[:3, joints.index(joint)] = linear_velocity
                    if not position_only:
                        jacobian[3:6, joint.index(joint)] = angular_velocity
                        jacobian = self.velocity_jacobian_to_spatial_jacobian(jacobian, pose_current2root)

                current_link = current_link.parent
            jacobians.append(jacobian)
        return jacobians

    @staticmethod
    def velocity_jacobian_to_spatial_jacobian(jacobian, link_pose):
        adjoint = np.eye(6)
        adjoint[:3, 3:6] = skew_matrix(link_pose[:3, 3])
        return adjoint @ jacobian

    @classmethod
    def _build_children(cls, parent_chain, current_mujoco_body):
        current_link = _link_from_mujoco_body(current_mujoco_body)
        parent_chain.add_child(current_link)
        for child in current_mujoco_body.body:
            cls._build_children(current_link, child)

    @classmethod
    def build_from_mjcf(cls, xml_string):
        import warnings
        mujoco_model = from_xml_string(xml_string)
        if len(mujoco_model.worldbody.body) > 1:
            warnings.warn(f"MuJoCo World Body has {len(mujoco_model.worldbody.body)} bodies. "
                          f"However this script will only use the first one")
        mujoco_root_body = mujoco_model.worldbody.body[0]
        root_link = _link_from_mujoco_body(mujoco_root_body)
        for child in mujoco_root_body.body:
            cls._build_children(root_link, child)
        return root_link


def _inv_trans(pose):
    inv = np.eye(4)
    inv[:3, :3] = pose[:3, :3].T
    inv[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return inv


def _get_mujoco_body_pose(mujoco_body):
    pose = np.eye(4)
    quat = np.array([1, 0, 0, 0])
    pos = np.zeros(3)
    if isinstance(mujoco_body.pos, np.ndarray):
        pos = mujoco_body.pos
    if isinstance(mujoco_body.quat, np.ndarray):
        quat = mujoco_body.quat
    pose[:3, :3] = transforms3d.quaternions.quat2mat(quat)
    pose[:3, 3] = pos
    return pose


def _link_from_mujoco_body(mujoco_body):
    body = Body(mujoco_body.name, _get_mujoco_body_pose(mujoco_body))
    body_joints = []
    for joint in mujoco_body.joint:
        joint_type = joint.type or "revolute"
        if joint.pos is not None:
            if not np.allclose(joint.pos, np.zeros(3)):
                raise RuntimeError(f"Current only mujoco joint with zero axis is supported!")
        joint_axis = np.array([0, 0, 1])
        if isinstance(joint.axis, np.ndarray):
            joint_axis = joint.axis
        joint_name = joint.name
        body_joints.append(Joint(joint_name, joint_type, joint_axis))

    return KinematicChain(body, body_joints)


if __name__ == '__main__':
    from hand_imitation.env.utils.mjcf_utils import string_to_array
    from hand_imitation.env.utils.mjcf_utils import xml_path_completion
    import time

    np.set_printoptions(precision=4)

    # MJCF Parsing
    path = xml_path_completion("adroit/test_adroit_kinematics.xml")
    with open(path, "r") as f:
        xml = f.read()
    chain = KinematicChain.build_from_mjcf(xml)
    print(chain.get_variable_joints())
    print(chain.forward_kinematics(np.zeros(25))['lftip'])

    # Forward Kinematics
    qpos = "-0.01; 0.015; 1.4; 0.5; 0.51; 0.16; -0.29; -0.0062; 9.6e-06; 0.29; 0.071; 0.0041; 9.7e-06; 0.29; 0.072;" \
           " 0.0041; 9.5e-06; 0.29; 0.072; 0.0041; 0.46; -0.075; 0.03; 0.031; 0.0027"
    qpos = string_to_array(qpos.replace(";", ""))
    pose = chain.forward_kinematics(qpos)['lftip']
    gt_pos = np.array([0.032896, -0.366942, 0.018367])
    assert np.allclose(gt_pos, pose[:3, 3], rtol=1e-3, atol=1e-5)

    # Jacobian
    init_qpos = np.random.rand(25)
    link_name = "mfmiddle"
    tic = time.time()
    jacobians = chain.jacobian(init_qpos, [link_name])[0]
    print(f"Jacobian computation takes {time.time() - tic}s")

    tic = time.time()
    numerical_jacobian = np.zeros([3, 25])
    forward_pose = chain.forward_kinematics(init_qpos)[link_name]
    for i in range(25):
        delta_qpos = init_qpos.copy()
        delta_qpos[i] += 1e-5
        velocity = (chain.forward_kinematics(delta_qpos)[link_name][:3, 3] - forward_pose[:3, 3]) / 1e-5
        numerical_jacobian[:, i] = velocity
    print(f"Numerical Jacobian computation takes {time.time() - tic}s")

    # Numerical Jacobian should align with real jacobian
    print(jacobians)
    print(np.linalg.norm(numerical_jacobian - jacobians))
