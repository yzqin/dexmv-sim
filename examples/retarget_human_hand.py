import argparse
import os
import pickle

import numpy as np
from natsort import natsorted

from hand_imitation.env.utils.mjcf_utils import xml_path_completion
from hand_imitation.kinematics.retargeting_optimizer import ChainMatchingPositionKinematicsRetargeting


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand_dir", type=str)
    parser.add_argument("--output_file", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    np.set_printoptions(precision=4)
    path = xml_path_completion("adroit/adroit_relocate.xml")
    link_names = ["palm", "thmiddle", "ffmiddle", "mfmiddle", "rfmiddle", "lfmiddle", "thtip", "fftip", "mftip",
                  "rftip", "lftip"][:6]
    solver = ChainMatchingPositionKinematicsRetargeting(path, link_names, has_joint_limits=True,
                                                        has_global_pose_limits=False)
    target_joint_index = [0, 2, 6, 10, 14, 18, 4, 8, 12, 16, 20][:6]

    hand_pose_files = natsorted(
        [os.path.join(args.hand_dir, file) for file in os.listdir(args.hand_dir) if
         file.endswith("npy") and "global" in file])
    hand_joint_files = natsorted(
        [os.path.join(args.hand_dir, file) for file in os.listdir(args.hand_dir) if
         file.endswith("npy") and "joint" in file])
    seq_len = min(len(hand_pose_files), len(hand_joint_files))

    results = []
    hand_frame_seq = []
    hand_joint_seq = []
    for i in range(seq_len):
        hand_frame = np.load(hand_pose_files[i])
        hand_joint = np.load(hand_joint_files[i])
        hand_frame_seq.append(hand_frame)
        hand_joint_seq.append(hand_joint)

    hand_joint_seq = np.stack(hand_joint_seq, axis=0)
    hand_frame_seq = np.stack(hand_frame_seq, axis=0)
    robot_joints = solver.retarget(hand_joint_seq[:, target_joint_index, :], hand_frame_seq,
                                   name="retargeting_example", verbose=True)

    with open(args.output_file, 'wb') as f:
        pickle.dump(robot_joints, f)


if __name__ == '__main__':
    main()
