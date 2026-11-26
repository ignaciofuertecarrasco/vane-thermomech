"""Quadratic tetrahedral meshes with gmsh, returned as NumPy arrays ready for CalculiX.

``volume_mesh`` meshes the current gmsh model (built from a STEP file or with the OCC kernel) and
returns the nodes, the C3D10 connectivity and, for every physical surface group, its boundary
triangles (6-node, corner nodes first). Node indices are 0-based and contiguous.

Quadratic elements are straight-sided (see ``volume_mesh``).

gmsh orders the mid-edge nodes of a 10-node tetrahedron (0-1, 1-2, 2-0, 0-3, 2-3, 1-3), Abaqus and
CalculiX (0-1, 1-2, 2-0, 0-3, 1-3, 2-3): the last two are swapped.
"""
from __future__ import annotations

from dataclasses import dataclass

import gmsh
import numpy as np

GMSH_TO_CCX_TET10 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]


@dataclass
class VolumeMesh:
    nodes: np.ndarray            # (n, 3)
    tets: np.ndarray             # (m, 10), CalculiX order, 0-based
    surfaces: dict               # name -> (k, 6) triangles, 0-based
    surface_entities: dict       # name -> list of gmsh surface tags

    @property
    def n_nodes(self):
        return len(self.nodes)

    def surface_nodes(self, name) -> np.ndarray:
        return np.unique(self.surfaces[name])


def volume_mesh(order: int = 2, optimise: bool = True) -> VolumeMesh:
    """Mesh the current model in 3D and collect the physical surface groups."""
    gmsh.model.mesh.generate(3)
    if optimise:
        gmsh.model.mesh.optimize("Netgen")      # on the linear mesh: cheap and improves the worst tetrahedra
    if order == 2:
        # Straight-sided quadratic elements: the mid-edge nodes stay at the edge midpoints instead of
        # being snapped to the CAD surface. Snapping curves the elements, and on the small passages and
        # the thin trailing edge some of them become so distorted (without being inverted) that the
        # stresses recovered at their nodes are meaningless, by orders of magnitude. With straight edges
        # every quadratic element is as valid as its linear parent; curved surfaces are faceted at the
        # scale of the element size, which the mesh study controls.
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
        gmsh.model.mesh.setOrder(2)
        if min_jacobian() <= 0:
            raise RuntimeError("quadratic mesh has elements with a non-positive Jacobian")
    tags, coords, _ = gmsh.model.mesh.getNodes()
    lookup = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    lookup[tags.astype(np.int64)] = np.arange(len(tags))
    nodes = coords.reshape(-1, 3)
    et, _, en = gmsh.model.mesh.getElements(3)
    k = list(et).index(11)                      # gmsh type 11 = 10-node tetrahedron
    tets = lookup[en[k].astype(np.int64)].reshape(-1, 10)[:, GMSH_TO_CCX_TET10]
    surfaces, entities = {}, {}
    for dim, ptag in gmsh.model.getPhysicalGroups(2):
        name = gmsh.model.getPhysicalName(dim, ptag)
        ents = list(gmsh.model.getEntitiesForPhysicalGroup(dim, ptag))
        tris = []
        for s in ents:
            t2, _, n2 = gmsh.model.mesh.getElements(2, s)
            for typ, nn in zip(t2, n2):
                if typ == 9:                    # 6-node triangle
                    tris.append(lookup[nn.astype(np.int64)].reshape(-1, 6))
        surfaces[name] = np.vstack(tris)
        entities[name] = ents
    # drop nodes not used by the tetrahedra (e.g. geometry points) and renumber
    used = np.unique(tets)
    if len(used) != len(nodes):
        remap = np.full(len(nodes), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        nodes, tets = nodes[used], remap[tets]
        surfaces = {k: remap[v] for k, v in surfaces.items()}
    return VolumeMesh(nodes, tets, surfaces, entities)


def min_jacobian() -> float:
    """Smallest scaled Jacobian over the 3D elements of the current gmsh mesh (must be > 0)."""
    _, tags, _ = gmsh.model.mesh.getElements(3)
    q = gmsh.model.mesh.getElementQualities(np.concatenate(tags), "minSJ")
    return float(q.min())


def tet_volumes(nodes, tets):
    a, b, c, d = (nodes[tets[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def triangle_areas_centroids(nodes, tris):
    a, b, c = (nodes[tris[:, i]] for i in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1), (a + b + c) / 3.0
