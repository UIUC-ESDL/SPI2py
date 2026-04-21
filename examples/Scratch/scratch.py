from __future__ import annotations
from collections import defaultdict, deque
from typing import Dict, Hashable, List, Tuple, Set
import networkx as nx
from networkx.algorithms.planarity import check_planarity

import numpy as np

def base_of(u: Hashable, long_to_base: Dict[Hashable, Hashable]) -> Hashable:
    return long_to_base.get(u, u)


def is_split_adj(u: Hashable, v: Hashable, long_to_base: Dict[Hashable, Hashable], G: nx.Graph | None = None) -> bool:
    """
    Decide whether (u,v) is a seam edge connecting split pieces.
    Prefer edge attr if available; otherwise infer by same base name.
    """
    if G is not None:
        data = G.get_edge_data(u, v, default={})
        if data.get("kind") == "split_adj":
            return True
    return base_of(u, long_to_base) == base_of(v, long_to_base)


def arc_excluding_endpoints(cycle: List[Hashable], a: Hashable, b: Hashable) -> List[Hashable]:
    """
    Given a cyclic list `cycle`, return the directed arc from a to b
    (walking forward in the list), excluding both endpoints.
    """
    n = len(cycle)
    ia = cycle.index(a)
    ib = cycle.index(b)

    out = []
    i = (ia + 1) % n
    while i != ib:
        out.append(cycle[i])
        i = (i + 1) % n
    return out


def unique_preserve_order(seq: List[Hashable]) -> List[Hashable]:
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def order_path_nodes(adj: Dict[Hashable, List[Hashable]]) -> List[Hashable]:
    """
    Order nodes in a seam adjacency subgraph that is a path (or a cycle).
    If it's a path, start at an endpoint. If it's a cycle, start anywhere.
    """
    nodes = list(adj.keys())
    endpoints = [u for u in nodes if len(adj[u]) == 1]
    start = endpoints[0] if endpoints else nodes[0]

    order = [start]
    prev = None
    cur = start
    while True:
        nxts = [x for x in adj[cur] if x != prev]
        if not nxts:
            break
        nxt = nxts[0]
        if nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
    return order


def compress_rotation_system(
    rot: Dict[Hashable, List[Hashable]],
    long_to_base: Dict[Hashable, Hashable],
    *,
    G_for_seams: nx.Graph | None = None,
) -> Dict[Hashable, List[Hashable]]:
    """
    Compress a rotation system by merging split nodes.

    rot: node -> CW neighbor cycle (must include seam neighbors if seam edges exist)
    long_to_base: split -> base mapping

    Returns: new_rot for the compressed graph (base nodes only)
    """
    # Group nodes by base
    groups: Dict[Hashable, List[Hashable]] = defaultdict(list)
    for u in rot.keys():
        groups[base_of(u, long_to_base)].append(u)
    groups = dict(groups)

    new_rot: Dict[Hashable, List[Hashable]] = {}

    for base, pieces in groups.items():
        if len(pieces) == 1 and pieces[0] == base:
            # Unsplit node: just map its neighbors to base-names and cleanup later
            mapped = [base_of(v, long_to_base) for v in rot[base] if base_of(v, long_to_base) != base]
            new_rot[base] = mapped
            continue

        # Build seam adjacency among pieces
        piece_set = set(pieces)
        seam_adj: Dict[Hashable, List[Hashable]] = {p: [] for p in pieces}
        for p in pieces:
            for v in rot[p]:
                if v in piece_set and is_split_adj(p, v, long_to_base, G_for_seams):
                    seam_adj[p].append(v)

        # Order the pieces along the seam
        order = order_path_nodes(seam_adj)

        merged_cycle: List[Hashable] = []
        for i, p in enumerate(order):
            seam_neighbors = seam_adj[p]  # 0,1,or2 (path endpoints have 1)

            # Endpoint: take full cycle without the single seam neighbor
            if len(seam_neighbors) == 1:
                s = seam_neighbors[0]
                seg = [x for x in rot[p] if x != s]
            elif len(seam_neighbors) == 2:
                a, b = seam_neighbors[0], seam_neighbors[1]
                # Choose the arc that corresponds to the "outside" of the split piece.
                # With only rotation info, there are two choices: arc(a->b) or arc(b->a).
                # We need consistent stitching along the path. A reliable rule is:
                #   for interior pieces, use arc(prev_piece -> next_piece) in the path order.
                prev_piece = order[i - 1] if i > 0 else None
                next_piece = order[i + 1] if i < len(order) - 1 else None
                if prev_piece is None or next_piece is None:
                    # Shouldn't happen for len==2 in a path, but just in case:
                    seg = arc_excluding_endpoints(rot[p], a, b)
                else:
                    seg = arc_excluding_endpoints(rot[p], prev_piece, next_piece)
            else:
                raise ValueError(
                    f"Piece {p} has seam-degree {len(seam_neighbors)}; expected 1 or 2. "
                    "Your seam graph is branching; need a different merge rule."
                )

            # Map neighbors to base names; drop internal-to-group
            seg_mapped = []
            for x in seg:
                bx = base_of(x, long_to_base)
                if bx == base:
                    continue
                seg_mapped.append(bx)

            merged_cycle.extend(seg_mapped)

        # Cleanup duplicates while preserving cyclic order as best as possible
        merged_cycle = unique_preserve_order(merged_cycle)
        new_rot[base] = merged_cycle

    # Final cleanup: remove self loops and ensure no duplicates remain
    for u, cyc in list(new_rot.items()):
        cyc2 = [v for v in cyc if v != u]
        new_rot[u] = unique_preserve_order(cyc2)

    return new_rot



def embedding_to_rot(emb):
    return {u: list(emb.neighbors_cw_order(u)) for u in emb}

def demo():
    # Same example as before
    G = nx.Graph()
    G.add_edges_from([("A","B"),("B","C"),("C","D"),("D","A")])
    G.add_edge("b0_0","A")
    G.add_edge("b0_1","B")
    G.add_edge("b0_2","C")
    G.add_edge("b0_2","D")
    G.add_edge("b0_0","b0_1", kind="split_adj")
    G.add_edge("b0_1","b0_2", kind="split_adj")

    long_to_base = {"b0_0":"b0","b0_1":"b0","b0_2":"b0"}

    is_planar, emb = check_planarity(G)
    assert is_planar
    rot = embedding_to_rot(emb)

    print("CW rotations (split):")
    for u in sorted(rot):
        print(u, ":", rot[u])

    new_rot = compress_rotation_system(rot, long_to_base, G_for_seams=G)

    print("\nCW rotations (compressed):")
    for u in sorted(new_rot):
        print(u, ":", new_rot[u])

if __name__ == "__main__":
    demo()
