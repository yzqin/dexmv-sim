import os
from typing import List

import numpy as np
from scipy import signal
from hand_imitation.misc.pose_utils import inverse_pose, project_rotation_to_axis


def get_default_joint_mapping():
    joint_mapping = {}
    joint_mapping.update(
        {"THJ4": (1, [0.707, 0, -0.707]), "THJ3": (1, [0.707, 0, 0.707]), "THJ2": (2, [0.707, 0, 0.707]),
         "THJ1": (2, [0, 1, 0]), "THJ0": (3, [0, 1, 0]), })
    joint_mapping.update(
        {"FFJ3": (4, [0, 1, 0]), "FFJ2": (4, [0, 0, 1]), "FFJ1": (5, [0, 0, 1]), "FFJ0": (6, [0, 0, 1]), })
    joint_mapping.update(
        {"MFJ3": (7, [0, 1, 0]), "MFJ2": (7, [-0.173, 0, 0.984]), "MFJ1": (8, [-0.173, 0, 0.984]),
         "MFJ0": (9, [-0.173, 0, 0.984]), })
    joint_mapping.update(
        {"RFJ3": (10, [0, 1, 0]), "RFJ2": (10, [-0.342, 0, 0.939]), "RFJ1": (11, [-0.342, 0, 0.939]),
         "RFJ0": (12, [-0.342, 0, 0.939]), })
    joint_mapping.update(
        {"LFJ4": (13, [-0.984, 0, 0.173]), "LFJ3": (13, [0, 1, 0]), "LFJ2": (13, [-0.642, 0, 0.766]),
         "LFJ1": (14, (-0.642, 0, 0.766)), "LFJ0": (15, [-0.642, 0, 0.766]), })
    return joint_mapping


def get_default_parent_mapping():
    joint_parent_mapping = {0: 0}
    joint_parent_mapping.update({1: 0, 2: 1, 3: 2})
    joint_parent_mapping.update({4: 0, 5: 4, 6: 5})
    joint_parent_mapping.update({7: 0, 8: 7, 9: 8})
    joint_parent_mapping.update({10: 0, 11: 10, 12: 11})
    joint_parent_mapping.update({13: 0, 14: 13, 15: 14})
    return joint_parent_mapping


def get_canonical_hand_joints():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    canonical_hand_file = os.path.join(current_dir, "canonical_hand_joints.npy")
    absolute_hand_frames = np.squeeze(np.load(canonical_hand_file))
    relative_hand_frames = np.ones_like(absolute_hand_frames)
    parent_mapping = get_default_parent_mapping()
    for child, parent in parent_mapping.items():
        relative_hand_frames[child] = inverse_pose(absolute_hand_frames[parent]) @ absolute_hand_frames[child]
    return relative_hand_frames


DEFAULT_JOINT_MAPPING = get_default_joint_mapping()
DEFAULT_PARENT_MAPPING = get_default_parent_mapping()
CANONICAL_HAND_JOINTS = get_canonical_hand_joints()


def get_robot_joint_pos_from_hand_frame(hand_frames: np.ndarray,
                                        robot_joint_names: List[str],
                                        ):
    # Parse hand frames into relative frame pose
    relative_hand_frames = np.zeros_like(hand_frames)
    for child, parent in DEFAULT_PARENT_MAPPING.items():
        relative_hand_frames[child] = inverse_pose(hand_frames[parent]) @ hand_frames[child]

    # Parse relative hand frames into joint pos
    robot_joint_pos = []
    for joint_name in robot_joint_names:
        hand_index, axis = DEFAULT_JOINT_MAPPING[joint_name]
        joint_pose_change = inverse_pose(CANONICAL_HAND_JOINTS[hand_index]) @ relative_hand_frames[hand_index]
        theta = project_rotation_to_axis(joint_pose_change[:3, :3], np.array(axis))
        robot_joint_pos.append(theta)
    return np.array(robot_joint_pos)


def visualize_hand_joint_frames(hand_frames: np.ndarray):
    import open3d
    if hand_frames.shape != (16, 4, 4):
        raise ValueError(f"Assume hand joint frame has shape 16x4x4, but got {hand_frames.shape}")
    root_joint_frame = hand_frames[0]
    root_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)
    root_frame.transform(root_joint_frame)
    frames = [root_frame]

    for i in range(1, hand_frames.shape[0]):
        frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        frame.transform(hand_frames[i])
        frames.append(frame)

    open3d.visualization.draw_geometries(frames)


def filter_position_sequence(position_seq: np.ndarray, wn=5, fs=25):
    sos = signal.butter(2, wn, 'lowpass', fs=fs, output='sos', analog=False)
    seq_shape = position_seq.shape
    if len(seq_shape) < 2:
        raise ValueError(f"Joint Sequence must have data with 3-dimension or 2-dimension, but got shape {seq_shape}")
    result_seq = np.empty_like(position_seq)
    if len(seq_shape) == 3:
        for i in range(seq_shape[1]):
            for k in range(seq_shape[2]):
                result_seq[:, i, k] = signal.sosfilt(sos, position_seq[:, i, k])
    elif len(seq_shape) == 2:
        for i in range(seq_shape[1]):
            result_seq[:, i] = signal.sosfilt(sos, position_seq[:, i])

    return result_seq


if __name__ == '__main__':
    project_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../")
    np.set_printoptions(precision=4)
    trivial_qpos = get_robot_joint_pos_from_hand_frame(CANONICAL_HAND_JOINTS, list(DEFAULT_JOINT_MAPPING.keys()))
    print("Trivial qpos of canonical pose relative to itself should be all zero")
    print(trivial_qpos)

    import transforms3d

    constant_offset = np.eye(4)
    constant_offset[:3, :3] = transforms3d.quaternions.quat2mat([0.707, 0.707, 0, 0])
    constant_offset[:3, 3] = np.random.rand(3)
    hand_joints_offset = np.matmul(constant_offset[None, :, :], CANONICAL_HAND_JOINTS)
    trivial_qpos_offset = get_robot_joint_pos_from_hand_frame(hand_joints_offset, list(DEFAULT_JOINT_MAPPING.keys()))

    assert np.allclose(trivial_qpos_offset,
                       trivial_qpos), "Hand joint qpos should not change during a fixed transformation"

    hand_file = os.path.join(project_root, "test_resources/hand_pose/relocation/mug/seq_0/0_hand/results_global_44.npy")
    hand_joint_frames = np.load(hand_file)
    qpos_original = get_robot_joint_pos_from_hand_frame(hand_joint_frames, list(DEFAULT_JOINT_MAPPING.keys()))
    hand_cam_to_object_cam = np.array(
        [[-0.0178, -0.7087, -0.7053, 0.4976], [0.7386, 0.4662, -0.487, 0.3935],
         [0.6739, -0.5296, 0.5151, 0.3154], [0., 0., 0., 1.]])
    hand_joint_frames = hand_cam_to_object_cam[None, ...] @ hand_joint_frames
    qpos_transform = get_robot_joint_pos_from_hand_frame(hand_joint_frames, list(DEFAULT_JOINT_MAPPING.keys()))
    print("Difference between original result and result for transformed pose", qpos_original - qpos_transform)
    print("Get qpos function should be invariant to rigid transformation.")

    project_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../")
    hand_file = os.path.join(project_root, "hand_imitation/misc/canonical_hand_joints.npy")
    hand_joint_frames = np.load(hand_file)
    visualize_hand_joint_frames(hand_joint_frames[0])
