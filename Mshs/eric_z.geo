// Gmsh project created on Fri Nov 22 17:13:27 2019
SetFactory("OpenCASCADE");
//+
Box(1) = {-0.24, -0.25, 0, 0.5, 0.5, 0.04};
//+
Physical Surface(1) = {1, 3, 4, 5, 2};
//+
Physical Surface(2) = {6};
