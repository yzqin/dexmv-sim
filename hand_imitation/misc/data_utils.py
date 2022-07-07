import glob
import os
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
from hand_imitation.misc.pose_utils import interpolate_transformation

OBJECT2ID = {'bleach_cleanser': '12', 'drill': '15', 'mug': '14', 'large_clamp': '19', 'sugar_box': '3',
             'mustard_bottle': '5', 'banana': '10', 'bowl': '13', 'cheez': '2', 'tomato_soup_can': '4'}
ID2OBJECT = {value: key for key, value in OBJECT2ID.items()}


def load_hand_object_data(hand_dir,
                          object_dir,
                          target_joint_index,
                          extrinsic=np.eye(4)) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
    joint_files = glob.glob(os.path.join(hand_dir, "joints_*.npy"))

    num_samples = len(joint_files)
    object_seq = []
    joint_seq = []
    for i in range(1, num_samples + 1):
        joint_file = os.path.join(hand_dir, f"joints_{i}.npy")
        object_file = os.path.join(object_dir, f"{i}.npy")
        joint_data = np.load(joint_file)
        object_data = np.load(object_file, allow_pickle=True)

        joint_data_world = extrinsic[:3, :3] @ joint_data[target_joint_index, :].T + extrinsic[:3, 3:4]
        joint_seq.append(joint_data_world.T)

        if object_data.size < 12:
            object_seq.append(None)
        else:
            # Compute world coordinate object pose
            object_pose_camera = np.concatenate([object_data, np.array([[0, 0, 0, 1]])], axis=0)
            object_data_world = (extrinsic @ object_pose_camera)
            object_seq.append(object_data_world)

    return joint_seq, object_seq


def load_hand_object_data_v2(hand_dir,
                             object_dir,
                             target_joint_index,
                             extrinsic=np.eye(4),
                             hand_cam_to_object_cam=None,
                             ) -> Tuple[List[np.ndarray], List[Dict[str, np.ndarray]], List[np.ndarray], int]:
    joint_files = glob.glob(os.path.join(object_dir, "*.npy"))
    num_samples = len(joint_files)
    object_seq = []
    joint_seq = []
    frame_seq = []
    missing_num = 0

    rotate_z_180 = np.eye(4)
    rotate_z_180[0, 0] = -1
    rotate_z_180[1, 1] = -1
    for i in range(1, num_samples):
        joint_file = os.path.join(hand_dir, f"joints_{i}.npy")
        frame_file = os.path.join(hand_dir, f"results_global_{i}.npy")
        object_file = os.path.join(object_dir, f"{i}.npy")

        # Skip the first frame if first frame is missing
        if (not os.path.exists(joint_file) or not os.path.exists(object_file)) and len(joint_seq) == 0:
            continue

        # Load object data and process
        if not os.path.exists(object_file):
            object_seq.append(object_seq[-1])
            missing_num += 1
        else:
            old_object_data: dict = np.load(object_file, allow_pickle=True).item()
            new_object_data = {}
            for index, pose in old_object_data.items():
                index_name = str(index).split(".")[0]
                if index_name not in ["10", "14"]:
                    continue
                object_name = ID2OBJECT[index_name]
                if pose.size < 12:
                    new_object_data[object_name] = None
                else:
                    # Compute world coordinate object pose
                    if pose.shape == (3, 4):
                        object_pose_camera = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
                    else:
                        object_pose_camera = pose
                    object_data_world = (extrinsic @ rotate_z_180 @ object_pose_camera)
                    new_object_data[object_name] = object_data_world

            object_seq.append(new_object_data)

        # Load joint data
        if not os.path.exists(joint_file):
            joint_seq.append(joint_seq[-1])
            frame_seq.append(frame_seq[-1])
            missing_num += 1
        else:
            # change extrinsic when hand and object is captured by different camera
            if hand_cam_to_object_cam is not None:
                hand_extrinsic = extrinsic @ hand_cam_to_object_cam
            else:
                hand_extrinsic = extrinsic

            joint_data = np.load(joint_file, allow_pickle=True)
            frame_data = np.load(frame_file, allow_pickle=True)
            joint_data_world = hand_extrinsic[:3, :3] @ joint_data[target_joint_index, :].T + hand_extrinsic[:3, 3:4]
            frame_data_world = np.matmul(hand_extrinsic[None, ...], frame_data)
            joint_seq.append(joint_data_world.T)
            frame_seq.append(frame_data_world)

    # Replace NaN and None
    joint_seq = replace_nan(joint_seq)
    frame_seq = replace_nan(frame_seq)
    return joint_seq, object_seq, frame_seq, missing_num


def replace_nan(array_seq: Union[np.ndarray, List[np.ndarray]]):
    if not np.isnan(np.sum(array_seq)):
        return array_seq
    else:
        nan_indices = []
        last_true_value = None
        for i in range(len(array_seq)):
            if np.isnan(np.sum(array_seq[i])):
                if last_true_value is not None:
                    array_seq[i] = last_true_value
                else:
                    nan_indices.append(i)
            else:
                last_true_value = array_seq[i]
                if len(nan_indices) > 0:
                    for k in range(len(nan_indices)):
                        array_seq[nan_indices[k]] = last_true_value
                    nan_indices.clear()

        if last_true_value is None:
            raise RuntimeError(f"Replace NaN fail because all data in sequence are NaN!")

        return array_seq


def interpolate_replay_sequence(retarget_qpos_seq: List[np.ndarray], object_pose_seq: List[Dict[str, np.ndarray]],
                                action_time_step, collect_time_step):
    action_len = int(((len(retarget_qpos_seq) - 1) * collect_time_step) // action_time_step + 1)
    object_pose_seq_new = [{}] * action_len
    retarget_qpos_seq_new = [None] * action_len

    # Pad one more data to avoid index exceed sequence length
    retarget_qpos_seq.append(retarget_qpos_seq[-1])
    object_pose_seq.append(object_pose_seq[-1])
    for i in range(action_len):
        first_num = int((i * action_time_step) // collect_time_step)
        residual = ((i * action_time_step) % collect_time_step) / collect_time_step
        retarget_qpos_seq_new[i] = retarget_qpos_seq[first_num] * (1 - residual) + retarget_qpos_seq[
            first_num + 1] * residual
        object_pose_new = {}
        for object_name, object_pose in object_pose_seq[first_num].items():
            trans = interpolate_transformation(object_pose, object_pose_seq[first_num + 1][object_name], 1 - residual)
            object_pose_new[object_name] = trans
        object_pose_seq_new[i] = object_pose_new
    return retarget_qpos_seq_new, object_pose_seq_new


def min_jerk_interpolate_replay_sequence(retarget_qpos_seq: List[np.ndarray],
                                         object_pose_seq: List[Dict[str, np.ndarray]], action_time_step,
                                         collect_time_step):
    from hand_imitation.misc.min_jerk import min_jerk
    seq_len = len(retarget_qpos_seq)
    action_len = int(((seq_len - 1) * collect_time_step) // action_time_step + 1)
    time_ratio = collect_time_step / action_time_step
    point_pass_time = np.arange(seq_len - 2) * time_ratio + time_ratio

    # Using Min Jerk to interpolate the joint trajectory
    retarget_qpos = np.stack(retarget_qpos_seq, axis=0)
    original_qpos_shape = retarget_qpos.shape
    retarget_qpos = np.reshape(retarget_qpos, (original_qpos_shape[0], -1))
    qpos, _, qvel, qacc = min_jerk(retarget_qpos, psg=point_pass_time, dur=action_len)
    for i in range(len(qpos)):
        qpos[i] = np.reshape(qpos[i], original_qpos_shape[1:])
        qvel[i] = np.reshape(qvel[i] / action_time_step, original_qpos_shape[1:])
        qacc[i] = np.reshape(qacc[i] / action_time_step ** 2, original_qpos_shape[1:])

    # Using linear method to interpolate object position
    object_pose_seq_new = [{}] * action_len
    object_pose_seq.append(object_pose_seq[-1])
    for i in range(action_len):
        first_num = int((i * action_time_step) // collect_time_step)
        residual = ((i * action_time_step) % collect_time_step) / collect_time_step
        object_pose_new = {}
        for object_name, object_pose in object_pose_seq[first_num].items():
            trans = interpolate_transformation(object_pose, object_pose_seq[first_num + 1][object_name], 1 - residual)
            object_pose_new[object_name] = trans
        object_pose_seq_new[i] = object_pose_new
    return qpos, qvel, qacc, object_pose_seq_new


def dict_has_none(data: Dict):
    if not data:
        return True
    has_none = False
    for key, value in data.items():
        if value is None:
            has_none = True
            break
    return has_none
