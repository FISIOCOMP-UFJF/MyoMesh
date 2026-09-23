//Mesh.CharacteristicLengthMax = 1;
//Mesh.CharacteristicLengthMin = 0.5;
//Mesh.CharacteristicLengthFactor = .4;

Geometry.HideCompounds = 0;
CharacteristicLengthFromCurvature = 1;
Lloyd = 1;
Mesh.Algorithm    = 6; // (1=MeshAdapt, 2=Automatic, 5=Delaunay, 6=Frontal, 7=bamg, 8=delquad) (Default=2)
Mesh.RecombineAll = 0;

Mesh.RemeshAlgorithm = 1; // (0=no split, 1=automatic, 2=automatic only with metis) (Default=0)


Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;

Mesh.SurfaceFaces = 1;

//Merge "/home/filipenamorato/mat2msh/output/20250220/Patient_1/stlFiles";


CreateTopology;
ll[] = Line "*";
L_LV_base = newl; Compound Line(L_LV_base) = ll[2];
L_RV_base = newl; Compound Line(L_RV_base) = ll[0];
L_epi_base = newl; Compound Line(L_epi_base) = ll[1];
Physical Line("EPI_BASE") = {L_epi_base};

ss[] = Surface "*";
S_LV = news; Compound Surface(S_LV) = ss[0];
S_RV = news; Compound Surface(S_RV) = ss[1];
S_epi = news; Compound Surface(S_epi) = ss[2];

LL_base = newll; Line Loop(LL_base) = {L_LV_base, L_RV_base, L_epi_base};
S_base  = news; Plane Surface(S_base) = {LL_base};

Physical Surface("BASE", 10) = {S_base};
Physical Surface("LV",   30) = {S_LV};
Physical Surface("RV",   20) = {S_RV};
Physical Surface("EPI",  40) = {S_epi};

SL_wall = newsl; Surface Loop(SL_wall) = {S_LV, S_RV, S_epi, S_base, 4};
V_wall  = newv;  Volume(V_wall)       = {SL_wall};
Physical Volume("WALL",  1)  = {V_wall};
