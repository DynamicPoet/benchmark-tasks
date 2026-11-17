// TPU95A stem; units mm. 100% infill, bend plane parallel to the bed,
// removable supports under the raised circular ends and flexible span.
// The spur cap holds a12.7x3mm steel disc in a globally cut pocket.
// The apple peg has a FLAT bonded steel end: a12.7mm disc cannot be
// recessed in the smaller12.6mm spigot. 8.9mm polymer +.1mm glue +3mm
// steel =12mm insertion length; largest inserted diameter12.7mm.
// Bonding and magnetic force values require the documented session checks;
// these dimensions are an engineering design, not measured calibration.

flex_d=6;
flex_len=20;
bend=19;
cap_d=15;
cap_h=4;
peg_d=12.6;
peg_h=12;
steel_d=12.7;
steel_h=3;
bond_h=.1;
cap_pocket_d=12.9;
cap_pocket_h=3.1;
$fn=64;

module apple_end_frame() {
    translate([cap_h+flex_len/2,0,0]) rotate([0,0,bend])
        translate([flex_len/2,0,0]) children();
}

module polymer() {
    difference() {
        union() {
            rotate([0,90,0]) cylinder(d=cap_d,h=cap_h);
            hull() {
                translate([cap_h,0,0]) sphere(d=flex_d);
                translate([cap_h+flex_len/2,0,0]) sphere(d=flex_d);
            }
            translate([cap_h+flex_len/2,0,0]) rotate([0,0,bend])
                hull() {sphere(d=flex_d);translate([flex_len/2,0,0]) sphere(d=flex_d);}
            apple_end_frame() rotate([0,90,0]) cylinder(d=peg_d,h=peg_h-steel_h-bond_h);
        }
        // Subtract AFTER union: the flexible root cannot refill this pocket.
        translate([-.01,0,0]) rotate([0,90,0]) cylinder(d=cap_pocket_d,h=cap_pocket_h+.01);
    }
}

polymer();
