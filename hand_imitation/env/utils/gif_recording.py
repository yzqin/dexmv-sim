import os
import time
from typing import List, Optional

import imageio
import numpy as np
import pygifsicle
from PIL import Image, ImageFont
from mujoco_py import MjRenderContextOffscreen

from hand_imitation.env.environments.base import MujocoEnv

_DefaultFont = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 30)


class RecordingGif:
    """
    This recorder is only used for evaluation. To visualize the training process, you may need to use other alternatives
    """

    def __init__(self, env: MujocoEnv, save_directory: str = "", camera_names: Optional[List[str]] = None, freq=4,
                 format="gif"):
        os.makedirs(save_directory, exist_ok=True)
        if not hasattr(env, "step"):
            raise AttributeError(f"The environment with class {env.__class__.__name__} do not have step attribute")
        if not hasattr(env, "reset"):
            raise AttributeError(f"The environment with class {env.__class__.__name__} do not have reset attribute")
        if env.has_renderer or env.viewer is not None:
            env.viewer = None
            env.has_offscreen_renderer = True

        if env.sim._render_context_offscreen is None:
            render_context = MjRenderContextOffscreen(env.sim)
            env.sim.add_render_context(render_context)
            env.sim._render_context_offscreen.vopt.geomgroup[2] = (1 if env.render_collision_mesh else 0)
            env.sim._render_context_offscreen.vopt.geomgroup[1] = (1 if env.render_visual_mesh else 0)

        env.has_renderer = True

        # Use default setting for resolution and set cameras
        self._env = env
        self._width = 1920
        self._height = 1080
        self.freq = freq
        self.format = format
        if not camera_names:
            self.camera_id = [self._env.mjpy_model.camera_name2id(camera_name) for camera_name in
                              self._env.mjpy_model.camera_names]
        else:
            self.camera_id = [self._env.mjpy_model.camera_name2id(camera_name) for camera_name in camera_names]

        # GIF Recording Setting
        self.directory = save_directory or '/tmp/hand_imitation'
        self.current_episode = 0
        self.video_queue = {cam_id: [] for cam_id in self.camera_id}
        timestamp = time.strftime('%m-%d_%H-%M-%S')
        env_name = env.__class__.__name__
        self.prefix = '{:s}.{:s}'.format(timestamp, env_name)

        # Save original function and modify it for gif recording features
        self.original_step_fn = env.step
        self.original_reset_fn = env.reset
        self.original_render_fn = env.render
        env.step = self._get_modified_step_fn()
        # env.reset = self._get_modified_reset_fn()
        env.render = self._get_modified_render_fn()

    def _get_modified_step_fn(self):
        _step = self._env.step
        _render_context = self._env.sim._render_context_offscreen
        _width = self._width
        _height = self._height
        _queue = self.video_queue

        def wrapped_step(action):
            obs, reward, done, info = _step(action)
            if self._env.timestep % self.freq == 0:
                for cam_id in self.camera_id:
                    _render_context.render(_width, _height, camera_id=cam_id)
                    data = _render_context.read_pixels(_width, _height, depth=False)
                    data = data[::-1, :, :]
                    _queue[cam_id].append(Image.fromarray(data))

            return obs, reward, done, info

        return wrapped_step

    def _get_modified_reset_fn(self):
        _reset = self._env.reset

        def wrapped_reset():
            obs = _reset()
            self.export(optimize_gif=True)
            self.current_episode += 1
            return obs

        return wrapped_reset

    def _get_modified_render_fn(self):
        _render = self._env.render
        _render_context = self._env.sim._render_context_offscreen
        _width = self._width
        _height = self._height
        _queue = self.video_queue

        def wrapped_render():
            if self._env.timestep % self.freq == 0:
                packed_images = []
                for cam_id in self.camera_id:
                    _render_context.render(_width, _height, camera_id=cam_id)
                    data = _render_context.read_pixels(_width, _height, depth=False)
                    data = data[::-1, :, :]
                    packed_images.append(Image.fromarray(data))

                combined = combine_multi_view(packed_images, {})
                _queue.append(combined)

        return wrapped_render

    def export(self, optimize_gif=True):
        if len(self.video_queue) == 0:
            return

        os.makedirs(self.directory, exist_ok=True)
        for cam_id in self.camera_id:
            if self.format == "gif":
                save_path = os.path.join(self.directory, f"{self.prefix}_{cam_id}_{self.current_episode:04}.gif")
                imageio.mimsave(save_path, self.video_queue[cam_id], 'GIF')
                if optimize_gif:
                    pygifsicle.optimize(save_path)
            elif self.format == "mp4":
                save_path = os.path.join(self.directory, f"{self.prefix}_{cam_id}_{self.current_episode:04}.mp4")
                imageio.mimsave(save_path, self.video_queue[cam_id], "mp4", fps=30)

            self.video_queue[cam_id].clear()
        print(f"Export video at {save_path}")
        timestamp = time.strftime('%m-%d_%H-%M-%S')
        env_name = self._env.__class__.__name__
        self.prefix = '{:s}.{:s}'.format(timestamp, env_name)

    def __enter__(self):
        self.current_episode = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if len(self.video_queue[self.camera_id[0]]) > 0:
            self.export(True)
        self._env.step = self.original_step_fn
        self._env.reset = self.original_reset_fn
        self._env.render = self.original_render_fn

    def clear(self):
        for cam_id in self.camera_id:
            self.video_queue[cam_id].clear()


def combine_multi_view(images: List[Image.Image], info: dict()):
    num_images = len(images)
    if num_images == 0:
        raise RuntimeError("No camera is specified for GIF Recording")
    num_row = int(np.ceil(np.sqrt(num_images)))
    image_size = images[0].size

    # Combine images from all cameras
    combined = Image.new('RGB', (num_row * image_size[0], num_row * image_size[1]))
    i = 0
    for i, image in enumerate(images):
        row = i // num_row
        column = i % num_row
        combined.paste(image, (column * image_size[0], row * image_size[1]))

    return combined
