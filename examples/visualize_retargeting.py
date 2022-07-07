import argparse
import os

import cv2
import numpy as np
import scipy.signal as signal
import transforms3d
from natsort import natsorted

from hand_imitation.kinematics.demonstration.relocation_demo import RelocationDemonstration

OBJECT2ID = {'bleach_cleanser': '12', 'drill': '15', 'mug': '14', 'large_clamp': '19', 'sugar_box': '3',
             'mustard_bottle': '5', 'banana': '10', 'bowl': '13', 'cheez': '2', 'tomato_soup_can': '4'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retargeting_result", type=str)
    parser.add_argument("--object_dir", type=str)
    return parser.parse_args()


def filter_joint_position(joint_seq: np.ndarray, wn=1, fs=300):
    sos = signal.butter(2, wn, 'lowpass', fs=fs, output='sos', analog=False)
    seq_shape = joint_seq.shape
    if len(seq_shape) < 2:
        raise ValueError(f"Pose sequence must have data with 3-dimension or 2-dimension, but got shape {seq_shape}")
    result_seq = np.empty_like(joint_seq)
    if len(seq_shape) == 3:
        for i in range(seq_shape[1]):
            for k in range(seq_shape[2]):
                result_seq[:, i, k] = signal.sosfilt(sos, joint_seq[:, i, k])
    elif len(seq_shape) == 2:
        for i in range(seq_shape[1]):
            result_seq[:, i] = signal.sosfilt(sos, joint_seq[:, i])
    return result_seq


def main():
    args = parse_args()
    np.set_printoptions(precision=4)
    skip_frame = 10  # skip initial frame to avoid bad estimation

    # Camera extrinsics
    # The extrinsic between two cameras can be computed using the same reference object, e.g. checkerboard
    hand_cam_to_object_cam = np.array(
        [[-0.0178, -0.7087, -0.7053, 0.4976], [0.7386, 0.4662, -0.487, 0.3935],
         [0.6739, -0.5296, 0.5151, 0.3154], [0., 0., 0., 1.]])
    object_extrinsic = np.linalg.inv(hand_cam_to_object_cam)

    # Load object pose and retargeted robot joints
    object_name = "mustard_bottle"
    object_id = OBJECT2ID[object_name]
    object_pose_files = natsorted(
        [os.path.join(args.object_dir, file) for file in os.listdir(args.object_dir) if file.endswith("npy")])[
                        skip_frame:]
    object_pose_list = [np.load(object_pose_file, allow_pickle=True)[()][object_id] for object_pose_file in
                        object_pose_files]
    retargeting_list = np.load(args.retargeting_result, allow_pickle=True)[skip_frame:]

    object_lie_list = [cv2.Rodrigues(object_extrinsic[:3, :3] @ object_pose[:3, :3])[0] for
                       object_pose in object_pose_list]
    object_pos_list = [object_extrinsic[:3, :3] @ object_pose[:3, 3] + object_extrinsic[:3, 3] for object_pose in
                       object_pose_list]

    # Filtering object pose
    init_pos = object_pos_list[0]
    init_lie = object_lie_list[0]
    object_pos_array_filter = filter_joint_position(np.stack(object_pos_list) - init_pos, wn=5, fs=100)
    object_lie_array_filter = filter_joint_position(np.stack(object_lie_list) - init_lie, wn=1, fs=500)
    object_pos_array_filter += init_pos
    object_lie_array_filter += init_lie
    object_quat_list_filter = [transforms3d.quaternions.mat2quat(cv2.Rodrigues(lie)[0]) for lie in
                               object_lie_array_filter]

    # Environment player
    player = RelocationDemonstration(has_renderer=True, object_name=object_name)

    # Visualization
    player.mjpy_model.geom_rgba[-9:-4] = np.zeros([5, 4])  # hide table for visualization purpose
    player.mjpy_model.geom_rgba[0] = np.zeros(4)  # hide floor for visualization purpose
    player.viewer.set_camera(0)

    # Hindsight target pose to be the same as last frame
    player.mjpy_model.body_pos[player.mjpy_model.body_name2id("target"), 0:3] = object_pos_list[-1]
    player.mjpy_model.body_quat[player.mjpy_model.body_name2id("target"), 0:4] = object_quat_list_filter[-1]

    data_len = min(len(retargeting_list), len(object_pose_list))
    player.filter.init_value(retargeting_list[0])
    dof = retargeting_list[0].shape[0]

    for i in range(data_len):
        object_pos = object_pos_array_filter[i]
        object_quat = object_quat_list_filter[i]
        robot_qpos = player.filter.next(retargeting_list[i])
        player.sim.data.qpos[:dof] = robot_qpos
        player.sim.data.qpos[player.object_trans_qpos_indices] = object_pos
        player.sim.data.qpos[player.object_rot_qpos_indices] = object_quat

        player.sim.forward()
        for _ in range(5):
            player.render()


if __name__ == '__main__':
    main()
