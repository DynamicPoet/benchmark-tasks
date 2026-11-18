"""Closed-solid mass integration for the declared reference-class secateurs.

The external catalogue fixes total mass and size, not individual part masses.
Steel/PVC densities and internal spine section are engineering estimates. Small
welded junction overlaps are retained in this gross-component estimate; the
catalogue total, rather than a dynamics fit, fixes the spine thickness once.
"""
from __future__ import annotations
import numpy as np

STEEL_DENSITY = 7850.0
PVC_DENSITY = 1400.0
CATALOGUE_MASS = 0.160
SPINE_WIDTH = 0.008
SPINE_FRONT, SPINE_BACK = -0.039, -0.129


def moments(vertices, faces):
    """Volume, first moment and second product moment of a closed triangle solid."""
    tri = np.asarray(vertices, float)[np.asarray(faces, int)]
    v = np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6
    sign = 1 if v.sum() > 0 else -1
    v *= sign
    sums = tri.sum(axis=1)
    first = np.einsum('i,ij->j', v, sums) / 4
    second = np.einsum('i,ijk,ijl->kl', v, tri, tri)
    second = (second + np.einsum('i,ij,ik->jk', v, sums, sums)) / 20
    volume = float(v.sum())
    if volume <= 0 or not np.isfinite(second).all():
        raise ValueError('Invalid closed-solid mass geometry')
    return volume, first, second


def material_properties(parts, spine_mesh_factory):
    """Integrate real component geometry; only internal steel section is inferred."""
    components = []
    for body, name, mesh in parts:
        density = PVC_DENSITY if name in ('grip', 'catch') else STEEL_DENSITY
        V, P, Q = moments(mesh.v, mesh.f)
        components.append((body, name, density * V, density * P, density * Q))
    base_mass = sum(c[2] for c in components)
    thickness = (CATALOGUE_MASS - base_mass) / (
        2 * (STEEL_DENSITY - PVC_DENSITY) * SPINE_WIDTH * (SPINE_FRONT - SPINE_BACK))
    if not 0.0005 <= thickness <= 0.004:
        raise ValueError(f'Catalogue mass incompatible with a plausible steel spine: {thickness}')
    for body, side in (('hook_half', -1), ('blade_half', 1)):
        mesh = spine_mesh_factory(side, thickness)
        V, P, Q = moments(mesh.v, mesh.f)
        # The sleeve's solid PVC estimate already occupies this exact volume.
        rho = STEEL_DENSITY - PVC_DENSITY
        components.append((body, 'steel_spine_replaces_PVC', rho*V, rho*P, rho*Q))
    result = {}
    for body in ('hook_half', 'blade_half', 'blade'):
        selected = [c for c in components if c[0] == body]
        mass = sum(c[2] for c in selected)
        p = sum((c[3] for c in selected), np.zeros(3))
        q = sum((c[4] for c in selected), np.zeros((3, 3)))
        com = p / mass
        inertia = np.trace(q)*np.eye(3) - q - mass*(np.dot(com,com)*np.eye(3)-np.outer(com,com))
        values, axes = np.linalg.eigh(inertia)
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        if values.min() <= 0 or values.max() > values.sum()-values.max()+1e-10:
            raise ValueError('Nonphysical inertia tensor')
        result[body] = dict(mass_kg=mass, com_m=com.tolist(), inertia_kg_m2=inertia.tolist(),
                            diagonal_inertia_kg_m2=values.tolist(), principal_matrix=axes.tolist())
    return dict(total_mass_kg=sum(r['mass_kg'] for r in result.values()),
                steel_density_kg_m3=STEEL_DENSITY, PVC_density_kg_m3=PVC_DENSITY,
                estimated_spine_width_m=SPINE_WIDTH, estimated_spine_thickness_m=thickness,
                components=[dict(body=b, name=n, mass_kg=m, com_m=(p/m).tolist()) for b,n,m,p,q in components],
                bodies=result)
