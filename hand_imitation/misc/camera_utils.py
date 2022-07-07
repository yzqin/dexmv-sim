import cv2
import numpy as np


def get_checkerboard_pose(image, board_size=(4, 5), square_size=0.03, intrinsic=np.eye(3), visualize=False):
    # Note that board_size = (a, b) and board_size = (b, a) will output different pose due to different origin
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(grey, board_size, None)
    if not found:
        import warnings
        warnings.warn("Checkerboard not found!")
        return None

    criteria_subpixel = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(grey, corners, (5, 5), (-1, -1), criteria_subpixel)
    obj_point = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    obj_point[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

    # Only for visualization
    if visualize:
        img = cv2.drawChessboardCorners(image, board_size, corners, True)
        cv2.imshow("checker_board", img)
        cv2.waitKey()

    _, rot_vector, trans_vector = cv2.solvePnP(obj_point, corners, intrinsic, None)
    rotation_matrix, _ = cv2.Rodrigues(rot_vector)
    homo_matrix = np.eye(4)
    homo_matrix[0:3, 0:3] = rotation_matrix
    homo_matrix[0:3, 3:4] = trans_vector
    return homo_matrix


def get_point_cloud_from_depth(depth_image, intrinsic, extrinsic=None):
    v, u = np.indices(depth_image.shape)  # [H, W], [H, W]
    z = depth_image  # [H, W]
    uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
    points_camera = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
    if extrinsic is None:
        return points_camera
    else:
        extrinsic_inv = np.linalg.inv(extrinsic)
        points_world = points_camera @ extrinsic_inv[:3, :3].T + extrinsic_inv[:3, 3]
        return points_world


def np2pcd(points, colors=None, normals=None):
    import open3d as o3d
    """Convert numpy array to open3d PointCloud."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.array(colors)
        if colors.ndim == 2:
            assert len(colors) == len(points)
        elif colors.ndim == 1:
            colors = np.tile(colors, (len(points), 1))
        else:
            raise RuntimeError(colors.shape)
        pc.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        assert len(points) == len(normals)
        pc.normals = o3d.utility.Vector3dVector(normals)
    return pc
