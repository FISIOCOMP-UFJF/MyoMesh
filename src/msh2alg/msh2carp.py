#!/usr/bin/env python
#
# Script to convert a mesh from Gmsh to CARP format
# Bernardo M. Rocha, 2014
#
# Patch 2025-07-17:
#   - Fixed .elem writing (correct type, 0-based reindex consistent with .pts, region = first tag or 1).
#   - Optional inclusion of LDRB fibers (mesh_fenics + fiber_fn/sheet_fn/normal_fn).
#   - split(" ") -> split() for more robust parsing.
#   - Added .lon (1 vector per element) keeping the rest legacy.
#
# Patch 2025-08-27:
#   - Exports ONLY tetrahedra (elem_type == 4) in .elem and generates .fib/.lon 1:1 with those TETs.
#   - .pts stays complete (no compaction).
#
import numpy as np
import os
import sys

CHANGE_BBOX = False  # keep legacy: if True, translates mesh to origin (careful!)

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
    assert ncomp == 3, "Vector field expected with 3 components."
    return v.reshape((-1, 3))

# ------------------------------------------------------------------ #
def gmsh2carp(gmshMesh, outputMesh,
              mesh_fenics=None,
              fiber_fn=None,
              sheet_fn=None,
              normal_fn=None,
              round_dec=8):
    """
    Convert Gmsh (.msh v2 ASCII) to CARP (.pts/.elem/.fib/.lon).
    Always exports ONLY TETs in .elem. The .fib/.lon will have one line per TET.
    The .pts stays complete (no compaction).
    """
    ptsFile  = outputMesh + '.pts'
    elemFile = outputMesh + '.elem'
    fibFile  = outputMesh + '.fib'
    lonFile  = outputMesh + '.lon'  # may not be written, depending on the case

    # ------------------------------------------------------------------
    # Read header (legacy: assumes fixed order)
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

    fpts = open(ptsFile, 'w')
    fpts.write(f"{num_nodes}\n")

    vpts   = np.zeros((num_nodes,3))
    id2idx = {}  # Gmsh node ID -> 0-based index used in .pts

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
    # Read elements
    # ------------------------------------------------------------------
    felem = open(elemFile,'w')
    line = f.readline()  # $Elements
    line = f.readline()  # num_elements

    num_elements = int(line)   
    elements = 0              # count of exported elements (TET only)
    nodes = []                # legacy list (not written; kept for debug)
    elem_types   = []         # element type after filter (always 4 = tetra)
    elem_nodes   = []         # Gmsh node ID tuple per element (after filter)
    elem_regions = []         # region tag (first tag, or 1 if absent)

    for i in range(num_elements):
        parts = f.readline().split()
        elem_id   = int(parts[0])
        elem_type = int(parts[1])
        num_tags  = int(parts[2])

        tags = []
        for j in range(num_tags):
            tags.append(int(parts[3+j]))
        region = tags[0] if tags else 1
        off = 3 + num_tags  # node data starts right after the tag list

        # Keep ONLY tetrahedra
        if elem_type == 4:  # tetra
            n1 = int(parts[off+0])
            n2 = int(parts[off+1])
            n3 = int(parts[off+2])
            n4 = int(parts[off+3])
            tup = (n1,n2,n3,n4)
        else:
            # unsupported type -> skip
            continue

        elements += 1
        nodes.append(tup)             # legacy
        elem_types.append(elem_type)  # always 4 here
        elem_nodes.append(tup)
        elem_regions.append(region)

    line = f.readline()  # $EndElements
    f.close()
    print(" Reading elements: Done.")

    print(" Number of nodes (file): %d" % num_nodes)
    print(" Number of elements (file): %d" % num_elements)
    print(" Exported TET elements: %d" % elements)

    # ------------------------------------------------------------------
    # Write .elem (TET only)
    # ------------------------------------------------------------------
    felem.write("%d\n" % (len(elem_types)))
    for t, nds, r in zip(elem_types, elem_nodes, elem_regions):
        nds0 = [id2idx[n] for n in nds]
        # t is always 4 here
        n1,n2,n3,n4 = nds0
        felem.write("Tt %d %d %d %d %d\n" % (n1,n2,n3,n4,r))
    felem.close()

    # ------------------------------------------------------------------
    # Write .fib and prepare .lon (1 line per exported TET)
    # ------------------------------------------------------------------
    ffib = open(fibFile,'w')

    use_ldrb = (mesh_fenics is not None and
                fiber_fn    is not None and
                sheet_fn    is not None and
                normal_fn   is not None)

    lon_lines = []  # fiber only (3 numbers) per TET element

    if not use_ldrb:
        for _ in range(len(elem_types)):
            ffib.write("%f %f %f %f %f %f %f %f %f\n" % (1,0,0,0,1,0,0,0,1))
            lon_lines.append((1.0,0.0,0.0))  # header=1 -> fiber only
        ffib.close()
        # write .lon keeping the simple "1" header
        with open(lonFile, "w") as flon:
            flon.write("1\n")
            for fval in lon_lines:
                flon.write("% .8e % .8e % .8e\n" % fval)
        return ptsFile, elemFile, fibFile, lonFile

    # ----- extract DG0 arrays -----
    f_arr = _extract_vec_dg0(fiber_fn)
    s_arr = _extract_vec_dg0(sheet_fn)
    n_arr = _extract_vec_dg0(normal_fn)

    # ----- FEniCS connectivity -----
    fen_cells  = mesh_fenics.cells()        # (Nc,4) if tetra
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
        # t is always 4 here and nds has 4 nodes
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

        ffib.write("%g %g %g %g %g %g %g %g %g\n" % (fval + sval + nval))
        lon_lines.append(fval)

    ffib.close()
    if misses:
        print(f" [gmsh2carp] Warning: {misses} tetra(s) with no fiber match; using identity.")

    # ---- write .lon (fiber only, 1 per TET) ----
    with open(lonFile, "w") as flon:
        flon.write("1\n")
        for fval in lon_lines:
            flon.write("% .8e % .8e % .8e\n" % fval)

    # sanity
    assert len(lon_lines) == len(elem_types), "lon size mismatch!"

    return ptsFile, elemFile, fibFile, lonFile


# ------------------------------------------------------------------ #
# (legacy)
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
