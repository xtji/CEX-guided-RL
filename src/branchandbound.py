from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any

import random

random.seed(123)


@dataclass
class Point:
    points: List[float]

@dataclass
class Interval:
    lb: float
    ub: float

    def contains(self, p: float):
        return self.lb <= p < self.ub

    @property
    def length(self):
        return self.ub - self.lb

    def __str__(self) -> str:
        return f'[{self.lb}, {self.ub}]'


@dataclass
class Box2D:
    bs: List[Interval]

    def contains(self, p: Point):
        return all(self.bs[i].contains(p.points[i]) for i in range(len(p.points)))

    def __str__(self) -> str:
        return f'Box({self.bs})'


class BranchAndBoundTree:

    def __init__(self, box: Box2D, children: Iterable[Box2D] = []) -> None:
        # TODO: information hiding
        self.box = box
        self.children = children

    def _str_aux(self, prefix):
        return prefix + str(self.box) + '\n' + \
               '\n'.join(c._str_aux(prefix + '  ') for c in self.children)

    def __str__(self) -> str:
        return self._str_aux('')

    def __repr__(self) -> str:
        return self.__str__()

def branch(box: Box2D, min_length: float):
    '''Splits a box always on the largest dimension, provided its length is less than the min_length'''
    # TODO: see if any smarter heuristics can do better to reduce number of boxes
    b1, b2 = None, None
    length = [box.bs[i].length for i in range(len(box.bs))]
    split_dim = length.index(max(length))
    if box.bs[split_dim].length > min_length:
        midpoint = round(box.bs[split_dim].lb + (box.bs[split_dim].length / 2), 2)
        b1 = Box2D([box.bs[i] if i != split_dim else Interval(box.bs[split_dim].lb, midpoint) for i in range(len(box.bs))])
        b2 = Box2D([box.bs[i] if i != split_dim else Interval(midpoint, box.bs[split_dim].ub) for i in range(len(box.bs))])
    return b1, b2

def bound(box: Box2D, points: Iterable[Point], min_digits):
    # TODO: Make more efficient by keeping points sorted
    points_within_box = [p for p in points if box.contains(p)]

    if len(points_within_box) == 0:
        return None

    bounding_box = Box2D([Interval(
            round(min(p.points[i] for p in points_within_box), 2),
            round(max(p.points[i] for p in points_within_box) + min_digits, 2)
        ) for i in range(len(box.bs))])
    return bounding_box

def false_positive_rate(box: Box2D, red_points: List[Point], blue_points: List[Point]) -> float:
    red_points_in_box = sum(box.contains(r) for r in red_points)
    blue_points_in_box = sum(box.contains(b) for b in blue_points)
    total_points_in_box = red_points_in_box + blue_points_in_box

    return red_points_in_box / total_points_in_box if total_points_in_box != 0 else 0

def expand_tree_bfs(node: BranchAndBoundTree, unsafe_points: Iterable[Point], safe_points: List[Point], min_digits: float, precision: float, min_length: float) -> Dict[int, Any]:
    depth = 0
    queue = [[node, depth]]
    inter_nodes = {}
    leaf_nodes = {}
    result = {}

    #ensure the size is less than max

    while queue:
        [current, current_depth] = queue.pop(0)
        if current_depth not in inter_nodes:
            inter_nodes[current_depth] = []
            leaf_nodes[current_depth] = []

        fp_safe_points = [r for r in safe_points if current.box.contains(r)]

        current.box.fpr = false_positive_rate(current.box, fp_safe_points, unsafe_points)
        if fp_safe_points and (current.box.fpr > (1 - precision)):
            b1, b2 = branch(current.box, min_length)
            if b1 is not None:
               b1 = bound(b1, unsafe_points, min_digits=min_digits)
            if b2 is not None:
                b2 = bound(b2, unsafe_points, min_digits=min_digits)
            if b1 is None and b2 is None:
                leaf_nodes[current_depth].append(current)
            else:
                if b1 is not None:
                    c1 = BranchAndBoundTree(b1, children=[])
                    current.children.append(c1)
                    queue.append([c1, current_depth + 1])
                if b2 is not None:
                    c2 = BranchAndBoundTree(b2, children=[])
                    current.children.append(c2)
                    queue.append([c2, current_depth + 1])
                inter_nodes[current_depth].append(current)
        else:
            leaf_nodes[current_depth].append(current)

    for depth, inter in inter_nodes.items():
        result[depth] = inter
        for i in range(depth+1):
            if leaf_nodes[i]:
                result[depth] += leaf_nodes[i]
        fprs = [b.box.fpr for b in result[depth]]
        avg_fpr = sum(fprs) / len(fprs)
        if avg_fpr <= (1 - precision):
            return result[depth]
    return result[depth]
