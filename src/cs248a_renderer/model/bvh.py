import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable
from collections import deque
import numpy as np
import slangpy as spy

from cs248a_renderer.model.bounding_box import BoundingBox3D
from cs248a_renderer.model.primitive import Primitive
from tqdm import tqdm


logger = logging.getLogger(__name__)


@dataclass
class BVHNode:
    # The bounding box of this node.
    bound: BoundingBox3D = field(default_factory=BoundingBox3D)
    # The index of the left child node, or -1 if this is a leaf node.
    left: int = -1
    # The index of the right child node, or -1 if this is a leaf node.
    right: int = -1
    # The starting index of the primitives in the primitives array.
    prim_left: int = 0
    # The ending index (exclusive) of the primitives in the primitives array.
    prim_right: int = 0
    # The depth of this node in the BVH tree.
    depth: int = 0

    def get_this(self) -> Dict:
        return {
            "bound": self.bound.get_this(),
            "left": self.left,
            "right": self.right,
            "primLeft": self.prim_left,
            "primRight": self.prim_right,
            "depth": self.depth,
        }

    @property
    def is_leaf(self) -> bool:
        """Checks if this node is a leaf node."""
        return self.left == -1 and self.right == -1


class BVH:
    def __init__(
        self,
        primitives: List[Primitive],
        max_nodes: int,
        min_prim_per_node: int = 1,
        num_thresholds: int = 16,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Builds the BVH from the given list of primitives using BREADTH-FIRST construction.
        This ensures balanced node distribution when max_nodes is limited.
        
        The build algorithm reorders the primitives in-place to align with the BVH node structure.
        At each node, the splitting axis and threshold are chosen using the Surface Area Heuristic (SAH)
        to minimize the expected cost of traversing the BVH during ray intersection tests.

        :param primitives: the list of primitives to build the BVH from
        :param max_nodes: the maximum number of nodes in the BVH
        :param min_prim_per_node: the minimum number of primitives per leaf node
        :param num_thresholds: the number of thresholds per axis to consider when splitting
        """
        self.nodes: List[BVHNode] = []
        self.primitives = primitives
        self.min_prim_per_node = min_prim_per_node
        self.on_progress = on_progress
        self.num_thresholds = num_thresholds
        self.max_nodes = max_nodes
        self.temp_idxs = []
        
        self.build_bvh()
        self.reorder_primitives()

    def build_bvh(self):
        
        if len(self.primitives) == 0:
            
            root = BVHNode(depth=0)
            self.nodes.append(root)
            self.temp_idxs.append([])
            return
        
        
        queue = deque()
        
        
        root_prim_idxs = list(range(len(self.primitives)))
        root = BVHNode(depth=0)
        root.bound = self.compute_bounding_box(root_prim_idxs)
        self.nodes.append(root)
        self.temp_idxs.append(root_prim_idxs)
        
        queue.append(0)
        
        while queue and len(self.nodes) < self.max_nodes:
            node_idx = queue.popleft()
            node = self.nodes[node_idx]
            prim_idxs = self.temp_idxs[node_idx]
            
            
            if len(prim_idxs) <=  self.min_prim_per_node:
                continue
            
            
            if len(self.nodes) >  self.max_nodes:
                continue
            
            best_split = self.find_best_split(prim_idxs, node.bound)
            
            if not  best_split:
                continue 
            

            left_idxs, right_idxs = self.bin_primitives(prim_idxs, best_split)
            
            if len(left_idxs) == 0 or len(right_idxs) == 0:
                continue
            

            left_node = BVHNode(depth=node.depth + 1)
            left_node.bound = self.compute_bounding_box(left_idxs)
            left_node_idx = len(self.nodes)
            self.nodes.append(left_node)
            self.temp_idxs.append(left_idxs)
            

            right_node = BVHNode(depth=node.depth + 1)
            right_node.bound =  self.compute_bounding_box(right_idxs)
            right_node_idx = len(self.nodes)
            self.nodes.append(right_node)
            self.temp_idxs.append(right_idxs)
            

            node.left = left_node_idx
            node.right = right_node_idx
            

            queue.append(left_node_idx)
            queue.append(right_node_idx)
            
            if self.on_progress:
                self.on_progress(len(self.nodes),  self.max_nodes)

    def compute_bounding_box(self, prim_idxs: List[int]) -> BoundingBox3D:

        if len(prim_idxs) == 0:
            return BoundingBox3D()
        
        bound = self.primitives[prim_idxs[0]].bounding_box

        for prim_idx in prim_idxs[1:]:
            bound = BoundingBox3D.union(bound, self.primitives[prim_idx].bounding_box)

        return bound

    def find_best_split(self, prim_idxs, bound):
        cost_min = float('inf')
        split_best = None

        for axis in [0, 1, 2]:
            min_ = bound.min[axis]
            max_ = bound.max[axis]

            thresholds = list(np.linspace(min_, max_, self.num_thresholds))

            for threshold in thresholds:
                left_box = BoundingBox3D()
                right_box = BoundingBox3D()
                left_count = 0
                right_count = 0

                for prim_idx in prim_idxs:
                    prim = self.primitives[prim_idx]
                    centroid = prim.bounding_box.center

                    if centroid[axis] < threshold:
                        left_box = BoundingBox3D.union(left_box, prim.bounding_box)
                        left_count += 1
                    else:
                        right_box = BoundingBox3D.union(right_box, prim.bounding_box)
                        right_count += 1

                if left_count == 0 or right_count == 0:
                    continue

                cost = self.get_sah_cost(
                    left_box, left_count, right_box, right_count, bound
                )

                if cost < cost_min:
                    cost_min = cost
                    split_best = (axis, threshold)

        leaf_cost = len(prim_idxs)
        if cost_min < leaf_cost:
            return split_best
        else:
            return None

    def bin_primitives(self, prim_idxs, split):
        axis, threshold = split

        left_idxs = []
        right_idxs = []

        for prim_idx in prim_idxs:
            prim = self.primitives[prim_idx]
            centroid = prim.bounding_box.center

            if centroid[axis] < threshold:
                left_idxs.append(prim_idx)
            else:
                right_idxs.append(prim_idx)
        return left_idxs, right_idxs

    def get_sah_cost(self, left_box, left_count, right_box, right_count, bound):
        sa_larger_box = bound.area

        if sa_larger_box <= 0:
            return float('inf')

        left_sa = left_box.area
        right_sa = right_box.area

        p1 = left_sa / sa_larger_box
        p2 = right_sa / sa_larger_box

        cost = p1 * left_count + p2 * right_count
        return cost

    def reorder_primitives(self):
        new_primitives = []

        def tree_in_order_trav(node_idx):
            node_cur = self.nodes[node_idx]

            if node_cur.left == -1:
                node_cur.prim_left = len(new_primitives)
                prim_idxs = self.temp_idxs[node_idx]
                for prim_idx in prim_idxs:
                    new_primitives.append(self.primitives[prim_idx])
                node_cur.prim_right = len(new_primitives)
            else:
                tree_in_order_trav(node_cur.left)
                tree_in_order_trav(node_cur.right)

        tree_in_order_trav(0)
        self.primitives[:] = new_primitives
        self.temp_idxs = []

            
def create_bvh_node_buf(module: spy.Module, bvh_nodes: List[BVHNode]) -> spy.NDBuffer:
    device = module.device
    node_buf = spy.NDBuffer(
        device=device, dtype=module.BVHNode.as_struct(), shape=(max(len(bvh_nodes), 1),)
    )
    cursor = node_buf.cursor()
    for idx, node in enumerate(bvh_nodes):
        cursor[idx].write(node.get_this())
    cursor.apply()
    return node_buf
