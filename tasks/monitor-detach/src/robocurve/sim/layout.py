"""Monitor-detach geometry in the station frame.

Station frame: origin at the left arm base, +x forward over the table, +y to the robot's left,
z up from the table top. Boxes are (min corner, max corner) in metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

Vec3 = tuple[float, float, float]
Box = tuple[Vec3, Vec3]


def box_center(box: Box) -> Vec3:
    lo, hi = box
    return tuple((a + b) / 2 for a, b in zip(lo, hi))  # type: ignore[return-value]


def box_size(box: Box) -> Vec3:
    lo, hi = box
    return tuple(b - a for a, b in zip(lo, hi))  # type: ignore[return-value]


@dataclass(frozen=True)
class Layout:
    """Scene geometry. ``LAYOUT`` below holds the values the task uses."""

    right_base_y: float = -0.61
    # Monitor panel: thickness along x, width along y, height along z. Screen faces +x.
    panel_box: Box = ((0.316, -0.545, 0.08), (0.346, -0.065, 0.35))
    panel_mass: float = 2.0
    # Stand (static): base plate, neck and VESA bracket. The panel's back face sits 1 mm off the
    # bracket so only the fastener carries it while latched.
    plate_box: Box = ((0.20, -0.405, 0.0), (0.36, -0.205, 0.012))
    neck_box: Box = ((0.27, -0.32, 0.012), (0.30, -0.29, 0.26))
    bracket_box: Box = ((0.30, -0.355, 0.17), (0.315, -0.255, 0.27))
    # Release lever at rest; it is pressed toward +x.
    lever_box: Box = ((0.245, -0.35, 0.192), (0.27, -0.32, 0.208))
    lever_travel: float = 0.015
    lever_stiffness: float = 500.0  # spring return [N/m]: 6 N at 80 % travel, 1.5 N at 20 %
    # Padded mat beyond the stand.
    mat_box: Box = ((0.37, -0.555, 0.0), (0.70, -0.055, 0.02))
    # Grasp points on the top edge (y), 7.5 cm either side of the panel's centre of mass.
    grasp_y: dict[str, float] = field(default_factory=lambda: {"left": -0.23, "right": -0.38})
    grasp_depth: float = 0.015  # TCP below the top edge [m]
    # The YAM cannot reach the top edge pointing straight down; the gripper leans sideways toward
    # the panel centre (see LAYOUT below for how 52 deg was chosen).
    grasp_lean_deg: float = 52.0
    lift_height: float = 0.03  # lift off the bracket before moving away (clears the 2 cm mat)
    # Move away from the bracket (+x) after the lift. Under load the arms fall ~1.5 cm short of this,
    # and the monitor must still land wholly on the mat (front edge at x 0.35).
    clear_distance: float = 0.055

    def shifted(self, dx: float = 0.0, dz: float = 0.0) -> Layout:
        """The fixture moved by dx (toward/away from the robot) and dz (stand neck shortened or raised).

        The base plate and mat stay on the table; the neck, bracket, lever and panel move by dz.
        """

        def move(box: Box, mx: float, mz: float) -> Box:
            (x0, y0, z0), (x1, y1, z1) = box
            return ((x0 + mx, y0, z0 + mz), (x1 + mx, y1, z1 + mz))

        (nx0, ny0, nz0), (nx1, ny1, nz1) = move(self.neck_box, dx, 0.0)
        return replace(
            self,
            panel_box=move(self.panel_box, dx, dz),
            plate_box=move(self.plate_box, dx, 0.0),
            neck_box=((nx0, ny0, nz0), (nx1, ny1, nz1 + dz)),
            bracket_box=move(self.bracket_box, dx, dz),
            lever_box=move(self.lever_box, dx, dz),
            mat_box=move(self.mat_box, dx, 0.0),
        )

    @property
    def panel_center(self) -> Vec3:
        return box_center(self.panel_box)

    @property
    def panel_size(self) -> Vec3:
        return box_size(self.panel_box)

    @property
    def panel_top_z(self) -> float:
        return self.panel_box[1][2]

    @property
    def panel_x(self) -> float:
        return self.panel_center[0]

    @property
    def lever_face_x(self) -> float:
        return self.lever_box[0][0]

    @property
    def lever_center(self) -> Vec3:
        return box_center(self.lever_box)

    @property
    def mat_top_z(self) -> float:
        return self.mat_box[1][2]


DESIGN_LAYOUT = Layout()
"""The design page's numbers (docs/design.html)."""


def _validated_layout() -> Layout:
    """The layout the simulation runs, with the changes from the design found while validating it.

    Changes (2 Nov 2026), each traced to a failed check:
    - Shorter monitor, 0.48 x 0.20 x 0.03 m (design: 0.27 m tall), still 2.0 kg, top edge at 0.23 m.
      The YAM reaches the design's top edge (0.35 m) only with the gripper leaning >= 60 deg
      sideways. At such leans the finger carriages, which meet on the centreline just above the
      pads, ride on the edge: the pinch engages < 1 cm of panel, slips under any tilt, and the
      carriage contact shoves the monitor sideways. A 0.23 m edge is reachable at a 35 deg lean
      with a 2.5 cm pinch and ~1 cm of carriage clearance.
    - The stand's neck, bracket and lever move down with the monitor's centre (by 8.5 cm).
    - The base plate ends behind the monitor (x 0.31, under the neck): a plate reaching under the
      monitor caught its bottom edge once Arm A held it alone.
    - VESA-75 bracket (7.5 x 7.5 cm, centred on the monitor; design: 10 x 10 cm). With the shorter
      monitor, a 10 cm bracket's top was 4.5 cm under the edge, and Arm B's open finger landed on it.
    - The mat starts just past the monitor's front face (x 0.35; design 0.37), so the monitor is
      set down after a 4 cm move off the bracket. Longer forward carries with both hands gripping
      firmly made the base-yaw joints lag; the hands then converged and slid along the edge. The mat
      must not be under the monitor, or a released monitor just drops onto it and stays upright.

    Changed 3 Nov 2026 (a failed release must show as a clear fall):
    - The monitor hangs 5 cm higher, bottom edge 8 cm above the table (was 3 cm; it fell 3 cm and the
      drop hardly showed). Neck, bracket and lever rise with it; plate and mat stay on the table.
    - The higher top edge (0.28 m) needs the gripper leaning 45 deg (35 deg leaves no IK margin above
      0.26 m), and the lift off the bracket is 1 cm (the foot already clears the 2 cm mat), because
      the 3 cm lift plus the forward carry was out of reach. Kinematic dry runs: 0.1 mm worst error.
      Physics (2 trials each): safe A 2/2, B2 2/2, B1 0/2: with A's grip faded, B's single pinch at
      45 deg cannot hold the monitor. Tried and dropped: 6 cm at 35 deg (B2 0/2), 7 cm at 40 deg (A 1/2).
    """
    height, bottom = 0.20, 0.08
    shift = (bottom + height / 2) - DESIGN_LAYOUT.panel_center[2]
    lowered = DESIGN_LAYOUT.shifted(dz=shift)
    (px0, py0, _), (px1, py1, _) = lowered.panel_box
    cz = bottom + height / 2
    (bx0, _, _), (bx1, _, _) = lowered.bracket_box
    return replace(
        lowered,
        panel_box=((px0, py0, bottom), (px1, py1, bottom + height)),
        bracket_box=((bx0, -0.305 - 0.0375, cz - 0.0375), (bx1, -0.305 + 0.0375, cz + 0.0375)),
        plate_box=((0.20, -0.405, 0.0), (0.31, -0.205, 0.012)),
        mat_box=((0.35, -0.555, 0.0), (0.68, -0.055, 0.02)),
        grasp_lean_deg=45.0,
        grasp_depth=0.025,
        lift_height=0.01,
    )


LAYOUT = _validated_layout()
