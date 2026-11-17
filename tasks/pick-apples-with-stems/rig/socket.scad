// Stem socket for an imitation apple (apple-picking practice rig). Print in PLA or PETG. Drill a16mm installation well30mm deep from the stem shoulder.
// Recess the socket mouth13mm below the shoulder, leaving an actual open well above it;
// glue the16.7mm socket in with epoxy. See apple-installation-section.scad. A 1/2 x 1/8 in N42 disc magnet
// (K&J Magnetics D82) is glued into the bottom of the bore. The stem's 12.6 mm peg slides in until its steel disc rests
// on the magnet; the bore's walls carry the stem's bending. Units: mm.

outer_d = 16;
bore_d = 12.8;
bore_depth = 12;     // the peg's length
magnet_d = 12.8;
magnet_h = 3.2;      // 1/8 in
floor_h = 1.5;
$fn = 64;

difference() {
    cylinder(d = outer_d, h = floor_h + magnet_h + bore_depth);
    translate([0, 0, floor_h]) cylinder(d = magnet_d, h = magnet_h + 0.01);
    translate([0, 0, floor_h + magnet_h]) cylinder(d = bore_d, h = bore_depth + 1);
}
