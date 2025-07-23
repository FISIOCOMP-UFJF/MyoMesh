#!/usr/bin/env python
#
# Script to convert a mesh from Gmsh to CARP format
# Bernardo M. Rocha, 2014
#
# Patch 2025-07-17:
#   - Corrigida escrita do .elem (tipo correto, reindex 0-based consistente com .pts, região = 1º tag ou 1).
#   - Inclusão opcional de fibras LDRB (mesh_fenics + fiber_fn/sheet_fn/normal_fn).
#   - split(" ") -> split() para parsing mais robusto.
#   - Adicionado .lon (1 vetor por elemento) mantendo o restante legado.
#
import numpy as np
import os
import sys

CHANGE_BBOX = False  # mantém legado: se True, translada mesh p/ origem (cuidado!)

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def bbox(coords):
    xmin,xmax = coords[:,0].min() , coords[:,0].max()
    ymin,ymax = coords[:,1].min() , coords[:,1].max()
    zmin,zmax = coords[:,2].min() , coords[:,2].max()
    print('[%f , %f] x [%f , %f] x [%f, %f]' % (xmin,xmax,ymin,ymax,zmin,zmax))
    return (xmin,xmax,ymin,ymax,zmin,zmax)

def _extract_vec_dg0(fn):
    """Convert FEniCS DG0 vector field -> (Nc,3) numpy array."""
    v = fn.vector().get_local()
    ncomp = fn.function_space().ufl_element().value_size()
    assert ncomp == 3, "Campo vetorial esperado com 3 componentes."
    return v.reshape((-1, 3))

# ------------------------------------------------------------------ #
def gmsh2carp(gmshMesh, outputMesh,
              mesh_fenics=None,
              fiber_fn=None,
              sheet_fn=None,
              normal_fn=None,
              round_dec=8):
    """
    Converter Gmsh (.msh v2 ASCII) para CARP (.pts/.elem/.fib/.lon).
    Se mesh_fenics & campos LDRB forem passados, escreve fibras reais nos TETRAs.
    Caso contrário, .fib = identidade (legado) e .lon não é criado.
    """
    ptsFile  = outputMesh + '.pts'
    elemFile = outputMesh + '.elem'
    fibFile  = outputMesh + '.fib'
    lonFile  = outputMesh + '.lon'  # pode não ser escrito, dependendo do caso

    # ------------------------------------------------------------------
    # Ler cabeçalho (legado: assume ordem fixa)
    # ------------------------------------------------------------------
    f = open(gmshMesh)

    # Skip everything until $EndMeshFormat is found
    for line in f:
        if line.strip() == "$EndMeshFormat":
            break

    # Now look for the $Nodes block (skipping PhysicalNames, Entities, etc)
    for line in f:
        if line.strip() == "$Nodes":
            break
    else:
        raise RuntimeError("Invalid .msh file: $Nodes block not found")

    num_nodes = int(f.readline().strip())

    # Começa a escrever o .pts
    fpts = open(ptsFile, 'w')
    fpts.write(f"{num_nodes}\n")

    vpts   = np.zeros((num_nodes,3))
    id2idx = {}  # mapa Gmsh node ID -> índice 0-based usado no .pts

    for i in range(num_nodes):
        parts = f.readline().split()
        node_id = int(parts[0])
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        vpts[i,0] = x
        vpts[i,1] = y
        vpts[i,2] = z
        id2idx[node_id] = i

    if CHANGE_BBOX:
        vb = bbox(vpts)
        vpts[:,0] = vpts[:,0] + (-1) * vb[0]
        vpts[:,1] = vpts[:,1] + (-1) * vb[2]
        vpts[:,2] = vpts[:,2] + (-1) * vb[4]

    print('Bounding box')
    vb = bbox(vpts)

    for i in range(num_nodes):
        x,y,z = vpts[i,:]
        fpts.write("%f %f %f\n" % (x,y,z))

    line = f.readline()  # $EndNodes
    fpts.close()
    print(" Reading nodes: Done.")
    
    # ------------------------------------------------------------------
    # Ler elementos
    # ------------------------------------------------------------------
    felem = open(elemFile,'w')
    line = f.readline()  # $Elements
    line = f.readline()  # num_elements

    num_elements = int(line)   
    elements = 0              # contador dos tipos suportados exportados
    nodes = []                # legado (não usado na escrita, mantido p/ debug)
    elem_types   = []         # tipo por elemento
    elem_nodes   = []         # tupla de IDs Gmsh por elemento
    elem_regions = []         # região (1º tag ou 1)

    for i in range(num_elements):
        parts = f.readline().split()
        elem_id   = int(parts[0])
        elem_type = int(parts[1])
        num_tags  = int(parts[2])

        tags = []
        for j in range(num_tags):
            tags.append(int(parts[3+j]))
        region = tags[0] if tags else 1
        off = 3 + num_tags #sempre começa em 3 no original, então coloquei off=3+num_tags

        if elem_type == 1:  # line
            n1 = int(parts[off+0])
            n2 = int(parts[off+1])
            tup = (n1,n2)
        elif elem_type == 2:  # tri
            n1 = int(parts[off+0])
            n2 = int(parts[off+1])
            n3 = int(parts[off+2])
            tup = (n1,n2,n3)
        elif elem_type == 5:  # hexa
            n1 = int(parts[off+0])
            n2 = int(parts[off+1])
            n3 = int(parts[off+2])
            n4 = int(parts[off+3])
            n5 = int(parts[off+4])
            n6 = int(parts[off+5])
            n7 = int(parts[off+6])
            n8 = int(parts[off+7])
            tup = (n1,n2,n3,n4,n5,n6,n7,n8)
        elif elem_type == 4:  # tetra
            n1 = int(parts[off+0])
            n2 = int(parts[off+1])
            n3 = int(parts[off+2])
            n4 = int(parts[off+3])
            tup = (n1,n2,n3,n4)
        else:
            # tipo não suportado -> ignora
            continue

        elements += 1
        nodes.append(tup)             # legado
        elem_types.append(elem_type)
        elem_nodes.append(tup)
        elem_regions.append(region)

    line = f.readline()  # $EndElements
    f.close()
    print(" Reading elements: Done.")

    print(" Number of nodes (file): %d" % num_nodes)
    print(" Number of elements (file): %d" % num_elements)
    print(" Exported elements: %d" % elements)

    # ------------------------------------------------------------------
    # Escrever .elem
    # ------------------------------------------------------------------
    felem.write("%d\n" % (len(elem_types)))
    for t, nds, r in zip(elem_types, elem_nodes, elem_regions):
        nds0 = [id2idx[n] for n in nds]
        if t == 1:
            n1,n2 = nds0
            felem.write("Ln %d %d %d\n" % (n1,n2,r))
        elif t == 2:
            n1,n2,n3 = nds0
            felem.write("Tr %d %d %d %d\n" % (n1,n2,n3,r))
        elif t == 5:
            n1,n2,n3,n4,n5,n6,n7,n8 = nds0
            felem.write("Hx %d %d %d %d %d %d %d %d %d\n" %
                        (n1,n2,n3,n4,n5,n6,n7,n8,r))
        elif t == 4:
            n1,n2,n3,n4 = nds0
            felem.write("Tt %d %d %d %d %d\n" % (n1,n2,n3,n4,r))
    felem.close()
    
    # ------------------------------------------------------------------
    # Escrever .fib e preparar .lon
    # ------------------------------------------------------------------
    ffib = open(fibFile,'w')

    use_ldrb = (mesh_fenics is not None and
                fiber_fn    is not None and
                sheet_fn    is not None and
                normal_fn   is not None)

    lon_lines = []  # só fibra (3 números) por elemento

    if not use_ldrb:
        for _ in range(len(elem_types)):
            ffib.write("%f %f %f %f %f %f %f %f %f\n" % (1,0,0,0,1,0,0,0,1))
            lon_lines.append((1.0,0.0,0.0))  # cabeçalho=1 -> só fibra
        ffib.close()
        # grava .lon mesmo assim? você decide. Aqui vou gravar para consistência:
        with open(lonFile, "w") as flon:
            flon.write("1\n")
            for fval in lon_lines:
                flon.write("% .8e % .8e % .8e\n" % fval)
        return ptsFile, elemFile, fibFile, lonFile

    # ----- extrair DG0 arrays -----
    f_arr = _extract_vec_dg0(fiber_fn)
    s_arr = _extract_vec_dg0(sheet_fn)
    n_arr = _extract_vec_dg0(normal_fn)

    # ----- conectividade FEniCS -----
    fen_cells  = mesh_fenics.cells()        # (Nc,4) se tetra
    fen_coords = mesh_fenics.coordinates()  # (Nf,3)

    fen_hash = {}
    for ci, c in enumerate(fen_cells):
        pts = fen_coords[c]
        key = tuple(sorted(tuple(np.round(pt, round_dec)) for pt in pts))
        fen_hash[key] = ci

    IDENT_F = (1.0,0.0,0.0)
    IDENT_S = (0.0,1.0,0.0)
    IDENT_N = (0.0,0.0,1.0)

    misses = 0
    for t, nds in zip(elem_types, elem_nodes):
        if t == 4 and len(nds) == 4:
            try:
                idxs = [id2idx[n] for n in nds]
                key  = tuple(sorted(tuple(np.round(vpts[idx], round_dec)) for idx in idxs))
                ci   = fen_hash.get(key)
            except Exception:
                ci = None

            if ci is None:
                misses += 1
                fval, sval, nval = IDENT_F, IDENT_S, IDENT_N
            else:
                fval = tuple(f_arr[ci])
                sval = tuple(s_arr[ci])
                nval = tuple(n_arr[ci])
        else:
            fval, sval, nval = IDENT_F, IDENT_S, IDENT_N

        ffib.write("%g %g %g %g %g %g %g %g %g\n" % (fval + sval + nval))
        lon_lines.append(fval)

    ffib.close()
    if misses:
        print(f" [gmsh2carp] Aviso: {misses} tetra(s) sem match de fibra; identidade usada.")

    # ---- escrever .lon (apenas fibra) ----
    with open(lonFile, "w") as flon:
        flon.write("1\n")
        for fval in lon_lines:
            flon.write("% .8e % .8e % .8e\n" % fval)

    # sanity
    assert len(lon_lines) == len(elem_types), "lon size mismatch!"

    return ptsFile, elemFile, fibFile, lonFile


# ------------------------------------------------------------------ #
# (legado)
# ------------------------------------------------------------------ #
def _main():
    if (len(sys.argv) < 3):
        print("\n Usage: gmsh2carp <gmsh_mesh> <carp_output>\n")
        sys.exit(-1)

    gmsh_mesh = sys.argv[1]

    if (not os.path.isfile(gmsh_mesh)):
        print("\n Error: the input gmsh %s does not exist.\n" % (gmsh_mesh))
        sys.exit(-1)

    gmsh2carp(sys.argv[1], sys.argv[2]) 

if __name__ == "__main__":
    _main()
