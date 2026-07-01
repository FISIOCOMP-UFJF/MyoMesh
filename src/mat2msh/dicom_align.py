"""
Aligns the pipeline mesh to the patient's DICOM coordinate space.

The required geometry (ImagePosition, ImageOrientation, resolution, thickness)
comes entirely from the Segment .mat file — identical values to the raw DICOMs
(validated: diff=0 for all fields). Only the .mat is needed.

The rigid transform (R, t) is computed once and applied to any surface/mesh file
produced by the pipeline (STL, PLY, MSH), mapping it from the pipeline's internal
frame to the real DICOM patient space (mm).

Pipeline flips accounted for in the transform:
  - saveMsh.py mirrors Z around the midpoint of [z_min, z_max] (invert_z=True by default)
  - the Segment row/column indexing (EndoX=row, EndoY=column) swaps two axes
    relative to the DICOM convention, so the optimal transform includes a reflection
    (det(R) = -1), recovered naturally by the SVD step (see _pixel_para_mundo).
  - makeSurface.py does NOT negate Y (the legacy intert_y flag was removed).

Note: computing R, t requires that barycenter regression (readMat.py) is DISABLED,
because per-slice shifts break the rigid relationship between the pipeline frame and
DICOM space. Therefore --align_dicom forces --no_regression in execAll.py.

Fix (2026-05-20): the pixel->world equation had px and py swapped.
Segment uses MATLAB convention (EndoX = row index, EndoY = column index), so
px must multiply v_coluna and py must multiply v_linha.
Validated with tools/diagnostico_pixel_mundo.py: median torso intensity goes from
168.7 (wrong) to 192.5 (correct). See ALINHAMENTO_DICOM.md.
"""

import os
import numpy as np
import meshio
from scipy.io import loadmat


def ler_geometria(mat_path):
    """Reads ImagePosition, ImageOrientation and pixel scales from the Segment .mat file."""
    s = loadmat(mat_path, simplify_cells=True)["setstruct"]
    ori = np.asarray(s["ImageOrientation"], dtype=float).flatten()
    v_linha  = ori[:3] / np.linalg.norm(ori[:3])
    v_coluna = ori[3:] / np.linalg.norm(ori[3:])
    geom = {
        "pos":       np.asarray(s["ImagePosition"], dtype=float).flatten(),
        "v_linha":   v_linha,
        "v_coluna":  v_coluna,
        "v_normal":  np.cross(v_linha, v_coluna),
        "res_x":     float(s["ResolutionX"]),
        "res_y":     float(s["ResolutionY"]),
        "espessura": float(s["SliceThickness"]) + float(s["SliceGap"]),
    }
    return geom, s


def _obter_pixels_contorno(s):
    """Extract pixelX, pixelY and slice index (0-based) from all contours."""
    listaX, listaY, listaFatia = [], [], []
    contornos = [("EndoX", "EndoY"), ("EpiX", "EpiY"),
                 ("RVEndoX", "RVEndoY"), ("RVEpiX", "RVEpiY")]
    for chaveX, chaveY in contornos:
        if chaveX not in s or chaveY not in s:
            continue
        X = np.asarray(s[chaveX], dtype=float)
        Y = np.asarray(s[chaveY], dtype=float)
        if X.ndim == 3 and X.shape[1] == 1:
            X, Y = X[:, 0, :], Y[:, 0, :]
        if X.ndim != 2 or X.size == 0:
            continue
        for fatia in range(X.shape[1]):
            mask = ~np.isnan(X[:, fatia]) & ~np.isnan(Y[:, fatia])
            if not mask.any():
                continue
            listaX.append(X[mask, fatia])
            listaY.append(Y[mask, fatia])
            listaFatia.append(np.full(mask.sum(), float(fatia)))
    return (np.concatenate(listaX),
            np.concatenate(listaY),
            np.concatenate(listaFatia))


def _pixel_para_mundo(px, py, fatia, geom):
    """
    DICOM equation: 1-based pixels -> mm coordinates in patient space.

    Segment stores the ImagePosition of the LAST DICOM slice (it reverses the
    slice order). So index fatia=0 corresponds to the last DICOM slice and the
    advance goes in the -v_normal direction (back to the first slice).

    Segment convention (MATLAB): EndoX is the ROW index (first matrix dimension,
    vertical image direction), EndoY is the COLUMN index (second dimension,
    horizontal direction). Therefore:
      px (EndoX) -> v_coluna (direction along columns = vertical)
      py (EndoY) -> v_linha  (direction along rows = horizontal)

    Validated against the Patient_2 DICOM torso: this convention gives a median
    intensity of 192.5 vs 168.7 for the inverse convention (px->v_linha, py->v_coluna).
    """
    return (geom["pos"]
            + (px[:, None] - 1) * geom["res_x"]     * geom["v_coluna"]
            + (py[:, None] - 1) * geom["res_y"]     * geom["v_linha"]
            - fatia[:, None]    * geom["espessura"] * geom["v_normal"])


def _frame_pipeline(px, py, fatia, geom):
    """
    Reconstruct the contour coordinates in the pipeline's internal frame.

    The only flip applied is the Z mirroring from saveMsh.py (invert_z=True by
    default); makeSurface.py no longer negates Y (the legacy intert_y flag was
    removed). The reflection seen in the fit (det(R)=-1) comes from the row/column
    axis swap in _pixel_para_mundo (Segment convention), not from this frame.
    """
    x = px * geom["res_x"]
    y = py * geom["res_y"]
    z = (fatia + 1) * geom["espessura"]
    z = z.min() + z.max() - z                    # saveMsh.py mirrors Z (invert_z default)
    return np.column_stack([x, y, z])


def _kabsch(origem, destino):
    """
    Orthogonal transform R and translation t minimizing ||R*origem + t - destino||.

    Does NOT force det(R)=+1: uses the raw R from the SVD. Since _pixel_para_mundo
    swaps the row/column axes (Segment convention), the optimal fit includes a
    reflection (det(R)=-1), recovered naturally by the SVD; RMSD~0.

    NOTE: RMSD~0 only attests self-consistency with _pixel_para_mundo/_frame_pipeline,
    not correctness vs the real DICOM. See WARNING at the top of the module.
    """
    c_o, c_d = origem.mean(0), destino.mean(0)
    U, _, Vt = np.linalg.svd((origem - c_o).T @ (destino - c_d))
    R = Vt.T @ U.T
    t = c_d - R @ c_o
    rms = float(np.sqrt((((origem @ R.T + t) - destino) ** 2).sum(1).mean()))
    return R, t, rms


def compute_dicom_transform(mat_path):
    """
    Compute the rigid transform from the pipeline frame to the DICOM space.

    Returns (R, t, rms): 3x3 rotation, (3,) translation and the fit RMSD in mm
    (should be ~0 when the regression is disabled).
    """
    geom, s = ler_geometria(mat_path)
    px, py, fatia = _obter_pixels_contorno(s)
    coord_mundo = _pixel_para_mundo(px, py, fatia, geom)
    frame_pipe  = _frame_pipeline(px, py, fatia, geom)
    R, t, rms = _kabsch(frame_pipe, coord_mundo)
    return R, t, rms


def apply_transform_to_file(path, R, t):
    """Apply R, t to the points of a mesh/surface file (STL, PLY, MSH) in place.

    If det(R) < 0 and the file contains triangles (STL/PLY), reverse the winding
    of each face to preserve the outward normal direction after the reflection.
    """
    ext = os.path.splitext(path)[1].lower()
    fmt = "gmsh" if ext == ".msh" else None
    mesh = meshio.read(path, file_format=fmt) if fmt else meshio.read(path)
    mesh.points = np.asarray(mesh.points, dtype=float) @ R.T + t
    if np.linalg.det(R) < 0:
        new_cells = []
        for block in mesh.cells:
            if "triangle" in block.type:
                conn = block.data.copy()
                conn[:, [1, 2]] = conn[:, [2, 1]]
                new_cells.append(meshio.CellBlock(block.type, conn))
            else:
                new_cells.append(block)
        mesh.cells = new_cells
    if ext == ".msh":
        meshio.write(path, mesh, file_format="gmsh22", binary=False)
    else:
        meshio.write(path, mesh)
