import glob
from xml.etree import ElementTree as ET

import numpy as np
import transforms3d

from hand_imitation.env.models.base import MujocoXML
from hand_imitation.env.utils.errors import ModelError
from hand_imitation.env.utils.mjcf_utils import xml_path_completion, array_to_string, find_elements

YCB_OBJECT_MAPPING = {"master_chef_can": "002", "cracker_box": "003", "sugar_box": "004", "tomato_soup_can": "005",
                      "mustard_bottle": "006", "potted_meat_can": "010", "banana": "011", "bleach_cleanser": "021",
                      "bowl": "024", "mug": "025", "scissors": "037", "large_clamp": "051", "foam_brick": "061"}
PRIMITIVE_TYPE = ["sphere", "capsule", "ellipsoid", "cylinder", "box"]
YCB_SIZE = {
    "master_chef_can": (0.1025, 0.1023, 0.1401),
    "cracker_box": (0.2134, 0.1640, 0.0717),
    "sugar_box": (0.0495, 0.0940, 0.1760),
    "tomato_soup_can": (0.0677, 0.0679, 0.1018),
    "mustard_bottle": (0.0576, 0.0959, 0.1913),
    "potted_meat_can": (0.0576, 0.1015, 0.0835),
    "banana": (0.1088, 0.1784, 0.0366),
    "bleach_cleanser": (0.1024, 0.0677, 0.2506),
    "bowl": (0.1614, 0.1611, 0.0550),
    "mug": (0.1169, 0.0930, 0.0813),
    "scissors": (0.09608, 0.201544, 0.015716),
    "large_clamp": (0.1659, 0.1216, 0.0364),
    "foam_brick": (0.05263, 0.077842, 0.051131),
}
YCB_ORIENTATION = {
    "master_chef_can": (1, 0, 0, 0),
    "cracker_box": (1, 0, 0, 0),
    "sugar_box": (1, 0, 0, 0),
    "tomato_soup_can": (1, 0, 0, 0),
    "mustard_bottle": (0.5, 0, 0, 0.866),
    "potted_meat_can": (1, 0, 0, 0),
    "banana": (1, 0, 0, 0),
    "bleach_cleanser": (0.707, 0, 0, 0.707),
    "bowl": (1, 0, 0, 0),
    "mug": (0.707, 0, 0, 0.707),
    "scissors": (1, 0, 0, 0),
    "large_clamp": (0, 0, 0, 1),
    "foam_brick": (1, 0, 0, 0),
}

VERSIONED_OBJECTS = ["mug"]


class PrimitiveObject:
    def __init__(self, primitive_name, primitive_size, free=True, density=1000, idn=0, **kwargs):
        if primitive_name not in PRIMITIVE_TYPE:
            raise ModelError(f"{primitive_name} is not a supported type for primitives")

        zero_pos = array_to_string([0, 0, 0])
        zero_quat = array_to_string([1, 0, 0, 0])
        self.body = ET.Element("body", name=f"{primitive_name}_{idn}", pos=zero_pos, quat=zero_quat)
        geom = ET.Element("geom", type=primitive_name, density=array_to_string(density), name=f"{primitive_name}_{idn}",
                          size=array_to_string(primitive_size), group="0", **kwargs)
        self.body.append(geom)

        if free:
            joint = ET.Element("joint", type='free', name=f"{primitive_name}_joint_{idn}", limited="false",
                               armature="0.001", frictionloss="0.001")
            self.body.append(joint)

    def merge_into_xml(self, mujoco_xml: MujocoXML, pos, quat):
        self.body.set("pos", array_to_string(pos))
        self.body.set("quat", array_to_string(quat))
        mujoco_xml.worldbody.append(self.body)

    @property
    def body_name(self):
        return self.body.get("name")

    @property
    def joint_name(self):
        joint = find_elements(root=self.body, tags="joint", return_first=True)
        return joint.get("name")


class YCBObject:
    def __init__(self, object_name, free=True, density=1000, idn=0, scale=1, **kwargs):
        object_full_name = f"{YCB_OBJECT_MAPPING[object_name]}_{object_name}"
        visual_file = xml_path_completion(f"ycb/visual/{object_full_name}/textured_simple_blender.msh")
        texture_file = xml_path_completion(f"ycb/visual/{object_full_name}/texture_map.png")
        v = kwargs.pop("version", "v0")
        object_has_version = object_name in VERSIONED_OBJECTS

        # Hack the mug for better simulation contact
        assets = list()
        mesh_scale = array_to_string([scale, scale, scale])
        if object_name == "mug" and v is "xml":
            self.body = build_mug(object_name, idn=idn, scale=scale, **kwargs)
        else:
            # Add model to rigid body
            zero_pos = array_to_string([0, 0, 0])
            zero_quat = array_to_string([1, 0, 0, 0])
            self.body = ET.Element("body", name=f"{object_name}_{idn}", pos=zero_pos, quat=zero_quat)
            if object_has_version:
                collision_files = sorted(glob.glob(xml_path_completion(f"ycb/collision/{object_full_name}/{v}/*.stl")))
                if len(collision_files) == 0:
                    raise RuntimeError(f"Object {object_name} has not collision mesh found with specified version {v}")
            else:
                collision_files = sorted(glob.glob(xml_path_completion(f"ycb/collision/{object_full_name}/*.stl")))
                if len(collision_files) == 0:
                    raise RuntimeError(
                        f"Object {object_name} has not collision mesh found, maybe this object must specify a version?")

            # Assets for both visual and collision mesh
            for i, collision_file in enumerate(collision_files):
                assets.append(
                    ET.Element('mesh', file=collision_file, name=f"{object_name}_collision_mesh_{i + 1}",
                               scale=mesh_scale))

            for i in range(len(collision_files)):
                geom = ET.Element("geom", type='mesh', mesh=f"{object_name}_collision_mesh_{i + 1}",
                                  name=f"collision_{object_name}_{i + 1}", density=array_to_string(density), group="2",
                                  **kwargs)
                self.body.append(geom)

        assets.append(ET.Element('mesh', file=visual_file, name=f"{object_name}_visual_mesh", scale=mesh_scale))
        assets.append(ET.Element('texture', file=texture_file, name=f"{object_name}_texture", type="2d"))
        assets.append(ET.Element('material', name=f"{object_name}_material", texture=f"{object_name}_texture"))
        self.assets = assets
        visual_geom = ET.Element('geom', mesh=f"{object_name}_visual_mesh", group="1", type="mesh", contype="0",
                                 conaffinity="0", material=f"{object_name}_material")
        self.body.append(visual_geom)

        # Add free joint if object is free
        if free:
            joint = ET.Element("joint", type='free', name=f"{object_name}_joint_{idn}", limited="false",
                               damping="0", frictionloss="0.001", armature="0.001")
            self.body.append(joint)

    def merge_into_xml(self, mujoco_xml: MujocoXML, pos, quat):
        self.body.set("pos", array_to_string(pos))
        self.body.set("quat", array_to_string(quat))

        mujoco_xml.worldbody.append(self.body)
        for asset in self.assets:
            if find_elements(root=mujoco_xml.asset, tags=asset.tag, attribs={"name": asset.get("name")},
                             return_first=True) is None:
                mujoco_xml.asset.append(asset)

    @property
    def body_name(self):
        return self.body.get("name")

    @property
    def joint_name(self):
        joint = find_elements(root=self.body, tags="joint", return_first=True)
        return joint.get("name")


def build_mug(object_name, density=1000, idn=0, scale=1, **kwargs):
    zero_pos = array_to_string([0, 0, 0])
    zero_quat = array_to_string([1, 0, 0, 0])
    body = ET.Element("body", name=f"{object_name}_{idn}", pos=zero_pos, quat=zero_quat)
    for i in range(12):
        size = array_to_string(np.array([0.028, 0.005, 0.08]) / 2 * scale)
        theta = np.deg2rad(i * 30)
        pos = array_to_string(np.array([0.045 * np.sin(theta) - 0.006, -0.042 * np.cos(theta), 0.0015]) * scale)
        quat = array_to_string(transforms3d.quaternions.axangle2quat(np.array([0, 0, 1]), theta))
        geom = ET.Element("geom", type='box', name=f"collision_{object_name}_{i + 1}", density=array_to_string(density),
                          group="2", size=size, pos=pos, quat=quat, **kwargs)
        body.append(geom)

    size = array_to_string(np.array([0.025, 0.012, 0.008]) / 2 * scale)
    pos1 = array_to_string(np.array([0.054, 0, 0.03]) * scale)
    pos2 = array_to_string(np.array([0.054, 0, -0.02]) * scale)
    quat1 = array_to_string(transforms3d.quaternions.axangle2quat(np.array([0, 1, 0]), np.deg2rad(40)))
    quat2 = array_to_string(transforms3d.quaternions.axangle2quat(np.array([0, 1, 0]), np.deg2rad(-40)))
    geom_handle1 = ET.Element("geom", type='box', name=f"collision_{object_name}_{13}",
                              density=array_to_string(density), group="2", size=size, pos=pos1, quat=quat1, **kwargs)
    geom_handle2 = ET.Element("geom", type='box', name=f"collision_{object_name}_{15}",
                              density=array_to_string(density), group="2", size=size, pos=pos2, quat=quat2, **kwargs)

    middle_size = array_to_string(np.array([0.008, 0.012, 0.04]) / 2 * scale)
    middle_pos = array_to_string(np.array([0.062, 0.00, 0.005]) * scale)
    geom_handle_middle = ET.Element("geom", type='box', name=f"collision_{object_name}_{14}",
                                    density=array_to_string(density), group="2", size=middle_size, pos=middle_pos,
                                    **kwargs)

    size_bottom = array_to_string([0.0425 * scale, 0.005 * scale])
    pos_bottom = array_to_string(np.array([0, 0, -0.033]) * scale)
    geom_bottom = ET.Element("geom", type='cylinder', name=f"collision_{object_name}_{16}",
                             density=array_to_string(density), group="2", size=size_bottom, pos=pos_bottom,
                             **kwargs)

    for geom in [geom_handle1, geom_handle2, geom_handle_middle, geom_bottom]:
        body.append(geom)
    return body


class SHAPENETObject:
    def __init__(self, object_name, free=True, density=1000, idn=0, scale=1, **kwargs):
        object_full_name = object_name
        visual_file = xml_path_completion(
            f"shapenet_mug/visual/{object_full_name}/model_transform_scaled.stl")

        collision_files = sorted(glob.glob(xml_path_completion(f"shapenet_mug/collision/{object_full_name}/*.stl")))
        if len(collision_files) == 0:
            raise RuntimeError(
                f"Object {object_name} has not collision mesh found, maybe this object must specify a version?")

        # Assets for both visual and collision mesh
        mesh_scale = array_to_string([scale, scale, scale])
        assets = list()
        assets.append(ET.Element('mesh', file=visual_file, name=f"{object_name}_visual_mesh", scale=mesh_scale))
        for i, collision_file in enumerate(collision_files):
            assets.append(
                ET.Element('mesh', file=collision_file, name=f"{object_name}_collision_mesh_{i + 1}", scale=mesh_scale))
        self.assets = assets

        # Add model to rigid body
        zero_pos = array_to_string([0, 0, 0])
        zero_quat = array_to_string([1, 0, 0, 0])
        self.body = ET.Element("body", name=f"{object_name}_{idn}", pos=zero_pos, quat=zero_quat)

        visual_geom = ET.Element('geom', mesh=f"{object_name}_visual_mesh", group="1", type="mesh", contype="0",
                                 conaffinity="0")
        self.body.append(visual_geom)
        for i in range(len(collision_files)):
            geom = ET.Element("geom", type='mesh', mesh=f"{object_name}_collision_mesh_{i + 1}",
                              name=f"collision_{object_name}_{i + 1}", density=array_to_string(density), group="2",
                              **kwargs)
            self.body.append(geom)

        # Add free joint if object is free
        if free:
            joint = ET.Element("joint", type='free', name=f"{object_name}_joint_{idn}", limited="false",
                               damping="0", frictionloss="0.001", armature="0.001")
            self.body.append(joint)

    def merge_into_xml(self, mujoco_xml: MujocoXML, pos, quat):
        self.body.set("pos", array_to_string(pos))
        self.body.set("quat", array_to_string(quat))

        mujoco_xml.worldbody.append(self.body)
        for asset in self.assets:
            if find_elements(root=mujoco_xml.asset, tags=asset.tag, attribs={"name": asset.get("name")},
                             return_first=True) is None:
                mujoco_xml.asset.append(asset)

    @property
    def body_name(self):
        return self.body.get("name")

    @property
    def joint_name(self):
        joint = find_elements(root=self.body, tags="joint", return_first=True)
        return joint.get("name")
