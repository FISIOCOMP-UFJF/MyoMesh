Mesh.CharacteristicLengthMax = 2;
Mesh.CharacteristicLengthMin = 1;

Geometry.HideCompounds = 0;
CharacteristicLengthFromCurvature = 1;
Lloyd = 1;
Mesh.Algorithm    = 6;
Mesh.RecombineAll = 0;

Mesh.RemeshAlgorithm = 1;

Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;

Mesh.SurfaceFaces = 1;

Printf("=== LV-ONLY GEO (STL, OLD-STYLE) LOADED ===");

// Import dos STLs acontece via linha de comando: gmsh -3 LVEndo.stl -merge LVEpi.stl lv_mesh.geo -o out.msh

CreateTopology;

// ------------------------------------------------------------
// Bases: mesmo estilo do antigo (Compound Line)
// ------------------------------------------------------------
ll[] = Line "*";
Printf("Lines found: %g", #ll[]);

// ATENCAO: a ordem depende do STL
// No teu antigo: ll[2]=LV_base, ll[1]=EPI_base (tinha RV também)
// Agora (LV-only) você vai precisar ajustar indices 1 vez olhando o log.
// Eu deixo no mesmo “formato” para você só trocar os indices se precisar.

If(#ll[] < 2)
  Printf("FATAL: expected >=2 lines for base loops, got %g", #ll[]);
  Abort;
EndIf

// Tentativa padrão: LV_base = ll[0], EPI_base = ll[1]
// Se ficar estranho, troque os indices aqui.
L_LV_base  = newl; Compound Line(L_LV_base)  = {ll[0]};
L_EPI_base = newl; Compound Line(L_EPI_base) = {ll[1]};

Physical Line("EPI_BASE") = {L_EPI_base};

// ------------------------------------------------------------
// Superficies: mesmo estilo do antigo (Compound Surface)
// ------------------------------------------------------------
ss[] = Surface "*";
Printf("Surfaces found: %g", #ss[]);

If(#ss[] < 2)
  Printf("FATAL: expected >=2 surfaces (endo+epi), got %g", #ss[]);
  Abort;
EndIf

// Tentativa padrão: ss[0]=LV endo, ss[1]=LV epi
// Se inverter, só trocar aqui.
S_LV_ENDO = news; Compound Surface(S_LV_ENDO) = {ss[0]};
S_LV_EPI  = news; Compound Surface(S_LV_EPI)  = {ss[1]};

// ------------------------------------------------------------
// Base em anel (não tampa o ventrículo)
// Loop externo = EPI, furo interno = ENDO
// ------------------------------------------------------------
LL_base = newll; Line Loop(LL_base) = {L_EPI_base, L_LV_base};
S_base  = news;  Plane Surface(S_base) = {LL_base};

Physical Surface("BASE", 10)     = {S_base};
Physical Surface("LV_ENDO", 30)  = {S_LV_ENDO};
Physical Surface("LV_EPI",  40)  = {S_LV_EPI};

// ------------------------------------------------------------
// Volume da parede (entre endo e epi, com base anelar)
// ------------------------------------------------------------
SL_wall = newsl; Surface Loop(SL_wall) = {S_LV_EPI, S_LV_ENDO, S_base};
V_wall  = newv;  Volume(V_wall)       = {SL_wall};

Physical Volume("WALL", 1)  = {V_wall};

Mesh 3;
