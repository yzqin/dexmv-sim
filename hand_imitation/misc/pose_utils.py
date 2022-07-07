import numpy as np
import transforms3d


def interpolate_rotation(mat1: np.ndarray, mat2: np.ndarray, mat1_weight: float):
    if mat1_weight < 0 or mat1_weight > 1:
        raise ValueError(f"Weight of rotation matrix should be 0-1, but given {mat1_weight}")

    relative_rot = mat1.T @ mat2
    # For numerical stability, first convert to quaternion and then to axis-angel for not-perfect rotation matrix
    axis, angle = transforms3d.quaternions.quat2axangle(transforms3d.quaternions.mat2quat(relative_rot))

    inter_angle = (1 - mat1_weight) * angle
    inter_rot = transforms3d.axangles.axangle2mat(axis, inter_angle)
    return mat1 @ inter_rot


def interpolate_transformation(mat1: np.ndarray, mat2: np.ndarray, mat1_weight: float):
    if mat1_weight < 0 or mat1_weight > 1:
        raise ValueError(f"Weight of rotation matrix should be 0-1, but given {mat1_weight}")

    result_pose = np.eye(4)
    rot1 = mat1[:3, :3]
    rot2 = mat2[:3, :3]
    inter_rot = interpolate_rotation(rot1, rot2, mat1_weight)
    inter_pos = mat1[:3, 3] * mat1_weight + mat2[:3, 3] * (1 - mat1_weight)
    result_pose[:3, :3] = inter_rot
    result_pose[:3, 3] = inter_pos
    return result_pose


def inverse_pose(pose: np.ndarray):
    inv_pose = np.eye(4, dtype=pose.dtype)
    inv_pose[:3, :3] = pose[:3, :3].T
    inv_pose[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return inv_pose


def project_rotation_to_axis(relative_rotation: np.ndarray, axis: np.ndarray):
    max_cos, max_theta = -1, 0
    for theta in np.linspace(0, 2 * np.pi, 360):
        rotation_guess = transforms3d.axangles.axangle2mat(axis, theta)
        trans = relative_rotation.T @ rotation_guess
        cos = (trans[0, 0] + trans[1, 1] + trans[2, 2] - 1) / 2
        if max_cos < cos:
            max_cos = cos
            max_theta = theta
    if max_theta > np.pi:
        max_theta -= 2 * np.pi

    return max_theta


def skew_matrix(vec):
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])


def pose2se3(pose: np.ndarray):
    axis, theta = transforms3d.axangles.mat2axangle(pose[:3, :3])
    skew = skew_matrix(axis)
    inv_left_jacobian = np.eye(3, dtype=np.float) / theta - 0.5 * skew + (
        1.0 / theta - 0.5 / np.tan(theta / 2)) * skew @ skew
    v = inv_left_jacobian @ pose[:3, 3]
    return np.concatenate([v, axis]), theta
