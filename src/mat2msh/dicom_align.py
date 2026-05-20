"""
Alinhamento da malha do pipeline ao espaço de coordenadas DICOM do paciente.

A geometria necessária (ImagePosition, ImageOrientation, resolução, espessura)
vem inteiramente do .mat do Segment — os mesmos valores que estão nos DICOMs
brutos (validado: diff=0 em todos os campos). Por isso só o .mat é necessário.

A transformação rígida (R, t) é calculada uma vez e aplicada a qualquer arquivo
de superfície/malha que o pipeline gere (STL, PLY, MSH), levando-o do frame
interno do pipeline para o espaço DICOM real (mm).

Flips fixos do pipeline considerados no cálculo:
  - makeSurface.py NÃO nega mais Y (intert_y = False) -> frame right-handed
  - saveMsh.py espelha Z no centro de [z_min, z_max] (invert_z=True por padrão)

Observação: o cálculo de R, t exige que a regressão de baricentros (readMat.py)
esteja DESLIGADA, pois ela aplica deslocamentos por fatia que quebram a relação
rígida entre o frame do pipeline e o espaço DICOM. Por isso --align_dicom força
--no_regression no execAll.py.

Correção (2026-05-20): a equação pixel→mundo estava com px↔py trocados.
O Segment usa convenção MATLAB (EndoX = índice de linha, EndoY = índice de coluna),
então px deve multiplicar v_coluna e py deve multiplicar v_linha.
Validado com tools/diagnostico_pixel_mundo.py: intensidade mediana no torso
passa de 168.7 (errado) para 192.5 (correto). Ver ALINHAMENTO_DICOM.md.
"""

import os
import numpy as np
import meshio
from scipy.io import loadmat


def ler_geometria(mat_path):
    """Lê ImagePosition, ImageOrientation e escalas do .mat do Segment."""
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
    """Extrai pixelX, pixelY e índice de fatia (0-based) de todos os contornos."""
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
    Equação DICOM: pixels 1-based → coordenadas mm no espaço do paciente.

    O Segment armazena ImagePosition da ÚLTIMA fatia DICOM (ele inverte a ordem
    das fatias). Por isso o índice fatia=0 corresponde à última fatia DICOM e o
    avanço se dá na direção -v_normal (de volta à primeira fatia).

    Convenção do Segment (MATLAB): EndoX é o índice de LINHA (primeira dimensão
    da matriz, direção vertical da imagem), EndoY é o índice de COLUNA (segunda
    dimensão, direção horizontal). Por isso:
      px (EndoX) → v_coluna (direção ao longo das colunas = vertical)
      py (EndoY) → v_linha  (direção ao longo das linhas = horizontal)

    Validado contra o torso DICOM do Patient_2: esta convenção dá intensidade
    mediana 192.5 vs 168.7 da convenção inversa (px→v_linha, py→v_coluna).
    """
    return (geom["pos"]
            + (px[:, None] - 1) * geom["res_x"]     * geom["v_coluna"]
            + (py[:, None] - 1) * geom["res_y"]     * geom["v_linha"]
            - fatia[:, None]    * geom["espessura"] * geom["v_normal"])


def _frame_pipeline(px, py, fatia, geom):
    """
    Reconstrói as coordenadas do contorno no frame interno do pipeline.

    Com makeSurface.py usando intert_y=False (Y não é mais negado), o único flip
    remanescente é o espelhamento de Z do saveMsh.py (invert_z=True por padrão).
    Sem a negação de Y, o frame é right-handed e a transformação para o DICOM
    sai como rotação pura (det=+1) em vez de reflexão.
    """
    x = px * geom["res_x"]
    y = py * geom["res_y"]
    z = (fatia + 1) * geom["espessura"]
    z = z.min() + z.max() - z                    # saveMsh.py espelha Z (invert_z padrão)
    return np.column_stack([x, y, z])


def _kabsch(origem, destino):
    """
    Transformação ortogonal R e translação t que minimizam ||R·origem + t − destino||.

    NÃO força det(R)=+1: usa o R puro do SVD. Com intert_y=False o frame fica
    right-handed e o ajuste sai como rotação pura (det=+1), RMSD≈0. (O R puro
    capturaria também uma reflexão, caso o frame voltasse a ser left-handed.)

    NOTA: RMSD≈0 só atesta autoconsistência com _pixel_para_mundo/_frame_pipeline,
    não correção vs DICOM real. Ver AVISO no topo do módulo.
    """
    c_o, c_d = origem.mean(0), destino.mean(0)
    U, _, Vt = np.linalg.svd((origem - c_o).T @ (destino - c_d))
    R = Vt.T @ U.T
    t = c_d - R @ c_o
    rms = float(np.sqrt((((origem @ R.T + t) - destino) ** 2).sum(1).mean()))
    return R, t, rms


def compute_dicom_transform(mat_path):
    """
    Calcula a transformação rígida do frame do pipeline para o espaço DICOM.

    Retorna (R, t, rms): rotação 3x3, translação (3,) e o RMSD do ajuste em mm
    (deve ser ~0 quando a regressão está desligada).
    """
    geom, s = ler_geometria(mat_path)
    px, py, fatia = _obter_pixels_contorno(s)
    coord_mundo = _pixel_para_mundo(px, py, fatia, geom)
    frame_pipe  = _frame_pipeline(px, py, fatia, geom)
    R, t, rms = _kabsch(frame_pipe, coord_mundo)
    return R, t, rms


def apply_transform_to_file(path, R, t):
    """Aplica R, t aos pontos de um arquivo de malha/superfície (STL, PLY, MSH) in place.

    Se det(R) < 0 e o arquivo contém triângulos (STL/PLY), reverte o winding de
    cada face para preservar a direção das normais externas após a reflexão.
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
