from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from pyglm import glm

from cs248a_renderer.model.transforms import Transform3D


# SceneObject index to generate unique IDs.
_scene_object_index = 0


def get_next_scene_object_index() -> int:
    global _scene_object_index
    idx = _scene_object_index
    _scene_object_index += 1
    return idx


@dataclass
class SceneObject:
    name: str = field(default_factory=lambda: f"object_{get_next_scene_object_index()}")
    transform: Transform3D = field(default_factory=Transform3D)
    # Parent SceneObject. None if this is the root object.
    parent: SceneObject | None = None
    children: List[SceneObject] = field(default_factory=list)

    def get_transform_matrix(self) -> glm.mat4:
        """Compute the world transform matrix by accumulating parent transforms.

        :return: The 4x4 world transform matrix.
        """

        #final = gml.identity(glm.vec(4))

        current = self.parent
        result = self.transform.get_matrix()

        while current:
            parent_m = current.transform.get_matrix()
            result = parent_m * result
            current = current.parent

        return result

        #for i in range(len(children)):
        #    final = glm.matmul(children[len(children)-i-1].transform.get_matrix(), final)

        #find the position from the final matrix

        #position = final[3].xyz
        #scale = glm.vec3(1.0, 1.0, 1.0)

        # find th scale
        #scale.x = glm.distance(final[0].xyz)
        #scale.y = glm.distance(final[1].xyz)
        #scale.z = glm.distance(final[2].xyz)

        #rotation = glm.mat3(final[0].xyz / scale.x, final[1].xyz / scale.y, final[2].xyz / scale.z)
        #rotation_quat = glm.quat_cast(rotation)

        #self.transform.scale = scale
        #self.transform.position = position
        #self.transform.rotation = rotation_quat


        #return #final #self.transform.get_matrix()

        

    def desc(self, depth: int = 0) -> str:
        indent = "  " * depth
        desc_str = f"{indent}SceneObject(name={self.name}, transform={self.transform}, children=[\n"
        for child in self.children:
            desc_str += child.desc(depth + 1) + "\n"
        desc_str += f"{indent}])"
        return desc_str

    def __repr__(self):
        return self.desc()
