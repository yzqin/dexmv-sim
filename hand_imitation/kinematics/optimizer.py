from typing import List

import nlopt
import numpy as np
import torch

from hand_imitation.kinematics.kinematic_chain import KinematicChain


class KinematicsOptimizer:
    def __init__(self, xml_string):
        self.chain = KinematicChain.build_from_mjcf(xml_string)
        self.dof = self.chain.chain_dof
        self.opt = nlopt.opt(nlopt.LD_SLSQP, self.dof)

    def set_joint_limit(self, joint_limits: np.ndarray):
        if joint_limits.shape != (self.dof, 2):
            raise ValueError(f"Expect joint limits have shape: {(self.dof, 2)}, but get {joint_limits.shape}")
        self.opt.set_lower_bounds(joint_limits[:, 0].tolist())
        self.opt.set_upper_bounds(joint_limits[:, 1].tolist())

    def get_last_result(self):
        return self.opt.last_optimize_result()


class PositionOptimizer(KinematicsOptimizer):
    def __init__(self, xml_string: str, body_names: List[str], huber_delta=0.01):
        super().__init__(xml_string)
        self.body_names = body_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)

        xml_body_names = list(self.chain.get_link_dict().keys())
        for body_name in body_names:
            if body_name not in xml_body_names:
                raise ValueError(f"Body {body_name} given does not appear to be in MuJoCo XML.")

    def _get_objective_function(self, target_pos, last_qpos):
        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            link_poses = self.chain.forward_kinematics(x)
            body_poses = [link_poses[body_name] for body_name in self.body_names]
            body_pos = np.array([pose[:3, 3] for pose in body_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.from_numpy(body_pos)
            torch_body_pos.requires_grad_()
            torch_target_pos = torch.as_tensor(target_pos)
            torch_target_pos.requires_grad_(False)

            # Loss term for kinematics retargeting based on 3D position error
            position_diff = torch.norm(torch_body_pos - torch_target_pos, dim=1)
            huber_distance = self.huber_loss(position_diff, torch.zeros_like(position_diff))
            result = huber_distance.cpu().detach().numpy()

            if grad.size > 0:
                jacobians = self.chain.jacobian(x, self.body_names, link_poses)

                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]
                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * 1e-3 * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return float(result)

        return objective

    def retarget(self, target_pos, last_qpos=None, verbose=True):
        if last_qpos is None:
            last_qpos = np.zeros(self.dof)
        last_qpos = last_qpos.tolist()
        objective_fn = self._get_objective_function(target_pos, last_qpos)
        self.opt.set_min_objective(objective_fn)
        self.opt.set_ftol_abs(1e-5)
        qpos = self.opt.optimize(last_qpos)
        min_value = self.opt.last_optimum_value()
        if verbose:
            print(f"Last distance: {min_value}")
        return qpos


class PoseOptimizer(KinematicsOptimizer):
    def __init__(self, xml_string: str, body_names: List[str], huber_delta=0.01, rotation_weight=1):
        super().__init__(xml_string)
        self.body_names = body_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)
        self.rotation_weight = rotation_weight

        xml_body_names = list(self.chain.get_link_dict().keys())
        for body_name in body_names:
            if body_name not in xml_body_names:
                raise ValueError(f"Body {body_name} given does not appear to be in MuJoCo XML.")

    def _get_objective_function(self, target_pose):
        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            link_poses = self.chain.forward_kinematics(x)
            body_poses = [link_poses[body_name] for body_name in self.body_names]
            body_pos = np.array([pose[:3, 3] for pose in body_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.from_numpy(body_pos)
            torch_body_pos.requires_grad_()
            torch_target_pos = torch.as_tensor(target_pose)
            torch_target_pos.requires_grad_(False)

            diff = torch.norm(torch_body_pos - torch_target_pos, dim=1)
            huber_distance = self.huber_loss(diff, torch.zeros_like(diff))
            result = huber_distance.cpu().detach().numpy()

            if grad.size > 0:
                jacobians = self.chain.jacobian(x, self.body_names, link_poses, position_only=False)

                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]
                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad[:] = grad_qpos[:]

            return float(result)

        return objective


if __name__ == '__main__':
    from hand_imitation.env.utils.mjcf_utils import xml_path_completion
    from hand_imitation.env.utils.mjcf_utils import string_to_array
    import time

    np.set_printoptions(precision=4)

    # MJCF Parsing
    link_names = ["thtip", "fftip", "mftip", "rftip", "lftip"]
    path = xml_path_completion("adroit/test_adroit_kinematics.xml")
    with open(path, "r") as f:
        xml = f.read()
    optimizer = PositionOptimizer(xml, link_names)
    qpos = "-0.0027; 0.0027; 0.65; -1.5; -0.61; 2.8; -8.9; 9.1; -0.19; -0.42; -5.7; 28; -0.96; " \
           "-0.079; 2.1; 12; -0.69; -0.0097; 8.2; 5.3; 0.56; -1.3; 0.94; 2.9; 22"
    qpos = string_to_array(qpos.replace(";", ""))
    final_target = [[-0.04, -0.31, 0.14], [-0.025, -0.24, 0.12], [0.026, -0.28, 0.091], [0.045, -0.27, 0.083],
                    [0.044, -0.32, 0.1]]
    tic = time.time()
    computed_qpos = optimizer.retarget(final_target, last_qpos=np.zeros(optimizer.dof))
    print(f"Kinematics Retargeting computation takes {time.time() - tic}s")
    print(computed_qpos - qpos)
    print(f"It takes {time.time() - tic}s to compute retargeting")
