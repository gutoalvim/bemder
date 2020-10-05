Merge "topology - Copy.msh";
SetFactory("OpenCASCADE");
Rectangle(32) = {-3, 0, 0.9, 6, 6, 0};
//+
BooleanIntersection{ Surface{27}; Surface{25}; Surface{24}; Surface{31}; Surface{30}; Surface{28}; Surface{29}; Surface{23}; Surface{22}; Surface{21}; Surface{26}; Delete; }{ Surface{32}; Delete; }
//+
BooleanIntersection{ Surface{24}; Surface{32}; Curve{55}; Curve{56}; Curve{58}; Curve{57}; Delete; }{ Surface{21}; Surface{23}; Surface{26}; Surface{27}; Surface{31}; Surface{30}; Surface{22}; Surface{28}; Surface{29}; Surface{25}; Delete; }
//+
BooleanDifference{ Surface{23}; Surface{31}; Delete; }{ Surface{32}; Delete; }
