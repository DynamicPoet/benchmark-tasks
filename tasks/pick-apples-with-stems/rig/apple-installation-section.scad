// Installation drawing, not an imitation-apple manufacturing requirement.
// Coordinates: +z from apple centre toward stem. Dimensions in mm.
// A75mm imitation apple is drilled16mm diameter,30mm deep from the
// stem shoulder plane z33.5. The16x16.7 socket mouth is recessed13mm
// at z20.5; the remaining open counterbore lets the6mm stem pass freely.
// Installed polymer ends at z3.8, leaving.3mm bottom bonding allowance.
$fn=64;
outer_d=16;
socket_h=16.7;
socket_mouth_z=20.5;
shoulder_z=33.5;
drill_depth=30;

module apple_section() {
    difference() {
        scale([1,1,67/75]) sphere(d=75);
        translate([0,0,shoulder_z-drill_depth]) cylinder(d=16,h=drill_depth+2);
        translate([0,-50,-50]) cube([50,100,100]);
    }
}
color([.8,.15,.1]) apple_section();
color([.15,.15,.15]) translate([0,0,socket_mouth_z-socket_h])
    difference() {
        cylinder(d=outer_d,h=socket_h);
        translate([0,0,1.5]) cylinder(d=12.8,h=socket_h+1);
        translate([0,-50,-1]) cube([50,100,50]);
    }
