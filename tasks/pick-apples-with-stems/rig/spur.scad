// Breakaway fruiting spur for the apple-picking practice rig. Print in PLA or PETG, 100 % infill, lying on its side.
// It screws onto the 25 mm trunk dowel (one 4 x 40 mm wood screw through the centre hole) and holds a
// 1/2 x 1/4 in N42 disc magnet (K&J Magnetics D84) flush with its tip, glued in with epoxy.
// Units: mm. The spur's axis is +z, from the trunk's surface (z = 0) to the tip.

trunk_d = 25.4;      // trunk dowel
spur_len = 25;       // trunk surface to tip face
tip_d = 16;          // spur body: holds the 12.7 mm magnet
magnet_d = 12.8;     // pocket: 1/2 in magnet + 0.1 mm clearance
magnet_h = 6.35;     // 1/4 in
screw_d = 4.2;
$fn = 64;

difference() {
    union() {
        // saddle on the trunk
        translate([0, 0, -4]) cylinder(d = 22, h = 6);
        // spur body, tapering slightly like a woody spur
        cylinder(d1 = 19, d2 = tip_d, h = spur_len);
    }
    // the trunk itself (the saddle's concave face)
    translate([0, 0, -trunk_d / 2]) rotate([90, 0, 0]) cylinder(d = trunk_d, h = 40, center = true);
    // magnet pocket, flush with the tip
    translate([0, 0, spur_len - magnet_h]) cylinder(d = magnet_d, h = magnet_h + 1);
    // screw hole (head from the tip side, under the magnet)
    translate([0, 0, -20]) cylinder(d = screw_d, h = spur_len + 20 - magnet_h - 1);
    translate([0, 0, spur_len - magnet_h - 8]) cylinder(d = 8, h = 8);
}
