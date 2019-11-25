Merge "topology_cmplx.msh";
SetFactory("OpenCASCADE");
Sphere(1) = {-0.2, 2, 1.8, 0.5, -Pi/2, Pi/2, 2*Pi};
//+
Transfinite Surface {13};
//+
Extrude {{0, 1, 0}, {0, 0, 0}, Pi/4} {
  Curve{56}; 
}
