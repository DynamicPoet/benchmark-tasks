"""Native blade/plant contact observations and a finite, target-specific load test."""
from __future__ import annotations

import numpy as np


def array(value) -> np.ndarray:
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value).copy()


class BladeContacts:
    """One physical blade sensor, filtered against every actual plant rigid body."""
    def __init__(self, base):
        tool, bush = base.scene["secateurs"], base.scene["bush"]
        physical = {"pot", "stub_col", "cut_top", "cut_brown", "cut_green"}
        self.filters = [p for p in bush.root_physx_view.link_paths[0] if p.rsplit("/", 1)[-1] in physical]
        paths = list(tool.root_physx_view.link_paths[0])
        blade = [p for p in paths if p.endswith("/blade")]
        if len(blade) != 1 or len(self.filters) != len(physical):
            raise RuntimeError("Blade/plant rigid-body inventory is incomplete")
        self.view = tool._physics_sim_view.create_rigid_contact_view(
            blade, filter_patterns=[self.filters], max_contact_data_count=2048)
        if self.view.sensor_count != 1 or self.view.filter_count != len(self.filters):
            raise RuntimeError("Native blade/plant contact filters did not resolve")
        self.dt = float(base.cfg.sim.dt)
        self.sensor = blade[0]
        robot = base.scene["right_robot"]
        fingers = [p for p in robot.root_physx_view.link_paths[0]
                   if p.endswith("/tip_left") or p.endswith("/tip_right")]
        if len(fingers) != 2:
            raise RuntimeError("Both actual tool-hand fingers must resolve")
        self.fingers, self.tool_paths = fingers, paths
        self.handles = tool._physics_sim_view.create_rigid_contact_view(
            fingers, filter_patterns=[paths] * 2, max_contact_data_count=1024)
        if self.handles.sensor_count != 2 or self.handles.filter_count != len(paths):
            raise RuntimeError("Native finger/tool contact filters did not resolve")

    def sample_handles(self) -> list[dict]:
        force, point, normal, separation, count, start = map(array, self.handles.get_contact_data(self.dt))
        matrix=array(self.handles.get_contact_force_matrix(self.dt))
        if int(count.sum()) >= 1024 or not np.isfinite(matrix).all():
            raise RuntimeError("Finger/tool contact buffer is saturated")
        rows = []
        for i, j in np.argwhere(count > 0):
            a = int(start[i, j]); b = a + int(count[i, j])
            total=(force[a:b].reshape(-1,1)*normal[a:b]).sum(0)
            if np.linalg.norm(total-matrix[i,j])>max(1e-6,1e-4*np.linalg.norm(matrix[i,j])):
                raise RuntimeError('Finger/tool points disagree with current-step pair force')
            for k in range(a, b):
                rows.append({"finger": self.fingers[i], "tool_body": self.tool_paths[j],
                             "point_world_m": point[k].tolist(),
                             "force_on_finger_N": (float(force[k, 0]) * normal[k]).tolist(),
                             "separation_m": float(separation[k, 0])})
        return rows

    def sample(self) -> list[dict]:
        force, point, normal, separation, count, start = map(array, self.view.get_contact_data(self.dt))
        matrix = array(self.view.get_contact_force_matrix(self.dt))
        if int(count.sum()) >= 2048 or not np.isfinite(matrix).all():
            raise RuntimeError("Blade contact buffer is saturated or nonfinite")
        rows = []
        for j, path in enumerate(self.filters):
            begin = int(start[0, j]); end = begin + int(count[0, j])
            vectors = force[begin:end].reshape(-1, 1) * normal[begin:end]
            total = vectors.sum(axis=0)
            if np.linalg.norm(total - matrix[0, j]) > max(1e-6, 1e-4 * np.linalg.norm(matrix[0, j])):
                raise RuntimeError("Blade contact points disagree with the same-dt native pair force")
            for k in range(end - begin):
                rows.append({"sensor": self.sensor, "plant_body": path, "body": path.rsplit("/", 1)[-1],
                             "point_world_m": point[begin + k].tolist(), "force_on_blade_N": vectors[k].tolist(),
                             "signed_normal_N": float(force[begin + k, 0]),
                             "separation_m": float(separation[begin + k, 0])})
        return rows


def target_load(rows: list[dict], a, b, radius: float, pivot, axis, bodies: set[str]) -> dict:
    """Current normal load from the intended finite cane, not the tool's total gauge.

    Plant colliders have the same circular cross-section as the visible stem.
    The small allowance covers contact/float error, not a neighbouring object.
    Only normal forces opposing blade closure count.
    """
    a, b, pivot, axis = map(lambda p: np.asarray(p, float), (a, b, pivot, axis))
    d = b - a; length2 = float(d @ d)
    force = torque = 0.; points = []
    for row in rows:
        if row["body"] not in bodies or row["separation_m"] > 0.0005:
            continue
        p = np.asarray(row["point_world_m"]); f = np.asarray(row["force_on_blade_N"])
        u = float((p - a) @ d / max(length2, 1e-15))
        if not 0 <= u <= 1 or np.linalg.norm(p - (a + u * d)) > radius + 0.0005:
            continue
        magnitude = float(np.linalg.norm(f)); moment = -float(np.cross(p - pivot, f) @ axis)
        if magnitude <= 0 or moment <= 0:
            continue
        lever = moment / magnitude
        if lever < 0.005:
            continue
        force += magnitude; torque += moment
        points.append({**row, "closing_resistance_Nm": moment, "effective_lever_m": lever})
    return {"normal_load_N": force, "resisting_moment_Nm": torque,
            "effective_lever_m": torque / force if force else None, "points": points}
