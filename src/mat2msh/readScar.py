import os
import glob
import subprocess
import argparse
import warnings
from pathlib import Path
from collections import namedtuple, defaultdict

import numpy as np
from scipy.io import loadmat
from skimage.measure import find_contours
import meshio
import pyvista as pv

# ============================================================
# Estrutura para ROIs clássicas (setstruct.Roi)
# ============================================================

ROIEntry = namedtuple('ROIEntry', ['name', 'z', 'points'])


def readScar_roi(mat_filename):
    """
    Lê o setstruct.Roi (cicatriz clássica) e retorna lista de ROIEntry.
    (Código baseado na sua versão original.)
    """
    print(f"[ROI] Reading ROIs from: {mat_filename}")
    data = loadmat(mat_filename, struct_as_record=False, squeeze_me=True)

    roi_cell = data['setstruct'].Roi
    entries = []

    for idx, roi in enumerate(roi_cell):
        # Nome da ROI
        name = getattr(roi, 'Name', f'ROI-{idx+1}')
        if isinstance(name, np.ndarray):
            name = name[0] if name.size > 0 else f'ROI-{idx+1}'
        name = str(name)

        # Coordenadas
        X = getattr(roi, 'X', [])
        Y = getattr(roi, 'Y', [])
        Z = getattr(roi, 'Z', [])

        if not isinstance(Z, (list, np.ndarray)):
            Z = [Z]
            X = [X]
            Y = [Y]

        for i in range(len(Z)):
            z_val = int(np.atleast_1d(Z[i]).flat[0])
            x_arr = np.atleast_1d(X[i]).flatten()
            y_arr = np.atleast_1d(Y[i]).flatten()
            if x_arr.size == 0 or y_arr.size == 0:
                continue
            pts = list(zip(x_arr, y_arr))
            entries.append(ROIEntry(name, z_val, pts))
    return entries


def group_by_slice(entries):
    """
    Recebe lista de ROIEntry e retorna:
      { z: { roi_name: [ (x,y), ... ], ... }, ... }
    """
    fatias = defaultdict(lambda: defaultdict(list))
    for e in entries:
        fatias[e.z][e.name].extend(e.points)
    return fatias


def save_fatias_to_txt(fatias, shifts_x_file, shifts_y_file, output_dir):
    """
    Aplica shifts X/Y por fatia e salva "slice_<z>.txt"
    com "x_aligned y_aligned z".
    (Versão original adaptada.)
    """
    os.makedirs(output_dir, exist_ok=True)
    shifts_x = np.loadtxt(shifts_x_file)
    shifts_y = np.loadtxt(shifts_y_file)

    for z, roi_map in sorted(fatias.items()):
        sx = shifts_x[z] if 0 <= z < len(shifts_x) else 0
        sy = shifts_y[z] if 0 <= z < len(shifts_y) else 0
        fname = os.path.join(output_dir, f"slice_{z}.txt")
        with open(fname, 'w') as f:
            for pts in roi_map.values():
                for x, y in pts:
                    f.write(f"{x - sx} {y - sy} {z}\n")
        print(f"[ROI] Saved aligned slice {z} to {fname}")


def save_rois_extruded_to_txt(fatias, mat_filename, output_dir, num_layers=1):
    """
    Usa SliceThickness, SliceGap, ResolutionX/Y do setstruct
    e extruda cada ROI de z_base até z_top em 'num_layers' camadas.
    (Essa é a sua extrusão original, por ROI.)
    """
    data = loadmat(mat_filename)
    ss = data['setstruct']
    slice_thickness = float(ss['SliceThickness'][0][0][0][0])
    gap             = float(ss['SliceGap'][0][0][0][0])
    resolution_x    = float(ss['ResolutionX'][0][0][0][0])
    resolution_y    = float(ss['ResolutionY'][0][0][0][0])
    dz = slice_thickness + gap

    os.makedirs(output_dir, exist_ok=True)

    for z, roi_map in sorted(fatias.items()):
        for roi_name, points in roi_map.items():
            safe_name = roi_name.replace(" ", "_").replace("/", "_")
            fname = os.path.join(output_dir, f"roi_{safe_name}_z{z}.txt")
            with open(fname, 'w') as f:
                z_base = z * dz
                z_top  = z_base + dz
                for layer in range(num_layers + 1):
                    alpha = layer / num_layers if num_layers > 0 else 0.0
                    z_interp = z_base * (1 - alpha) + z_top * alpha
                    for x, y in points:
                        x_out = x * resolution_x
                        y_out = y * resolution_y
                        f.write(f"{x_out:.6f} {y_out:.6f} {z_interp:.6f}\n")
            print(f"[ROI] Saved extruded ROI '{roi_name}' (slice {z}) to: {fname}")


# ============================================================
# GreyZone.map → GZ/Core
# ============================================================

def load_gz_map_and_metadata(mat_path):
    """
    Lê o arquivo .mat (Segment) e retorna:
      - lbl: (H, W, S) com rótulos (0=bg, 1=GZ, 2=Core)
      - resolution_x, resolution_y (mm/pixel)
      - dz (mm) = SliceThickness + SliceGap
    """
    md = loadmat(mat_path, simplify_cells=True)

    try:
        ss = md["setstruct"]
        gz_map = ss["Scar"]["GreyZone"]["map"]
    except Exception as e:
        raise RuntimeError(
            "Não encontrei 'setstruct/Scar/GreyZone/map' no .mat."
        ) from e

    lbl = np.array(gz_map)
    if lbl.ndim == 2:
        lbl = lbl[..., None]
    if lbl.ndim != 3:
        raise ValueError(f"Esperava (H, W, S) para GreyZone.map, obtive ndim={lbl.ndim}")

    try:
        slice_thickness = float(ss["SliceThickness"])
        slice_gap       = float(ss["SliceGap"])
        resolution_x    = float(ss["ResolutionX"])
        resolution_y    = float(ss["ResolutionY"])
    except Exception as e:
        raise RuntimeError(
            "Problema ao ler SliceThickness/SliceGap/ResolutionX/ResolutionY em setstruct."
        ) from e

    dz = slice_thickness + slice_gap
    return lbl, resolution_x, resolution_y, dz


def extract_contours_all_slices(lbl, value, dx=1.0, dy=1.0, invert_y=False):
    """
    lbl: (H, W, S) labels
    value: rótulo alvo (1 = Greyzone, 2 = Core)
    dx, dy: espaçamento mm/pixel
    Retorna: lista de tamanho S; cada item é lista de polígonos (N,2) em (x_mm, y_mm)
    """
    H, W, S = lbl.shape
    all_slices = []
    for s in range(S):
        mask = (lbl[:, :, s] == value).astype(float)
        cs = find_contours(mask, level=0.5)
        polys = []
        for c in cs:
            y, x = c[:, 0], c[:, 1]
            X = x * dx
            Y = (H - 1 - y) * dy if invert_y else y * dy
            polys.append(np.column_stack([X, Y]))
        all_slices.append(polys)
    return all_slices


def save_region_extruded_txts(polys_per_slice, dz, out_dir, region_name,
                              num_layers=1, z_offset=0.0):
    """
    Gera um arquivo .txt POR REGIÃO (polígono) em cada fatia.

    polys_per_slice: lista de tamanho S; cada item é lista de polígonos (N,2), em mm
    dz            : espaçamento entre fatias (mm)
    out_dir       : diretório onde salvar os .txt
    region_name   : nome curto da região, ex.: "greyzone" ou "core"
    num_layers    : número de camadas entre z_base e z_top (>=1)
    z_offset      : deslocamento absoluto em Z (mm)

    Saída:
      roi_<region_name>_sSSS_cCCC.txt
    """
    os.makedirs(out_dir, exist_ok=True)
    txt_paths = []

    for s, polys in enumerate(polys_per_slice):
        if not polys:
            continue

        z_base = z_offset + s * dz
        z_top  = z_base + dz

        for c_idx, poly in enumerate(polys):
            if poly.size == 0:
                continue

            fname = os.path.join(
                out_dir,
                f"roi_{region_name}_s{s:03d}_c{c_idx:03d}.txt"
            )
            txt_paths.append(fname)

            with open(fname, "w") as f:
                f.write(
                    f"# region={region_name}, slice={s}, "
                    f"component={c_idx}, points={len(poly)}\n"
                )
                for layer in range(num_layers + 1):
                    alpha = layer / num_layers
                    z = (1.0 - alpha) * z_base + alpha * z_top
                    for x, y in poly:
                        f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
                    f.write("\n")

            print(f"[GZ] {region_name} — slice {s}, comp {c_idx} salvo em {fname}")

    return txt_paths


# ============================================================
# Geração de superfícies PLY/STL (usado por ambos os modos)
# ============================================================

def generate_surfaces_and_stl(patient_id, rois_dir, ply_dir, stl_dir):
    """
    Para cada arquivo *.txt em rois_dir:
      1) makeSurface.py -> .ply
      2) PlyToStl       -> .stl
    """
    os.makedirs(ply_dir, exist_ok=True)
    os.makedirs(stl_dir, exist_ok=True)

    txts = sorted(glob.glob(os.path.join(rois_dir, "*.txt")))
    if not txts:
        warnings.warn(f"Nenhum arquivo .txt encontrado em {rois_dir}.")
        return

    for txt in txts:
        base = os.path.splitext(os.path.basename(txt))[0]
        ply = os.path.join(ply_dir, f"{base}.ply")
        stl = os.path.join(stl_dir, f"{base}.stl")

        surface_command = (
            f"python3 src/mat2msh/makeSurface.py {txt} "
            f"--output_dir {ply_dir} --patient_id {patient_id} --cover-both-ends"
        )
        try:
            subprocess.run(surface_command, shell=True, check=True)
            print(f"[PLY] Surface for {txt} generated at {ply}")
        except subprocess.CalledProcessError as e:
            print(f"[ERRO] Gerando PLY para {txt}: {e}")
            continue

        if os.path.exists(ply):
            try:
                cmd = (
                    f"./convertPly2STL/build/bin/PlyToStl {ply} {stl} "
                    f"1 1 0.0002 200 1"
                )
                subprocess.run(cmd, shell=True, check=True)
                print(f"[STL] STL created: {stl}")
            except subprocess.CalledProcessError as e:
                print(f"[ERRO] Convertendo {ply} para STL: {e}")
        else:
            print(f"[AVISO] PLY {ply} não encontrado. Pulando conversão STL.")


# ============================================================
# msh_tag_to_ply (usado no execAll.py)
# ============================================================

def msh_tag_to_ply(msh_path, tag=2, ply_path="fibrose_surface.ply"):
    """
    Lê um .msh com tetraedros e tags gmsh:physical,
    extrai os elementos com a tag especificada e
    salva a superfície correspondente em um .ply ASCII.
    """
    msh_path, ply_path = Path(msh_path), Path(ply_path)
    msh = meshio.read(msh_path)

    all_t, all_tags = [], []
    for i, cb in enumerate(msh.cells):
        if cb.type == "tetra":
            all_t.append(cb.data)
            all_tags.append(msh.cell_data["gmsh:physical"][i])
    if not all_t:
        raise ValueError("Sem tetraedros no .msh.")

    tets = np.vstack(all_t)
    tags = np.concatenate(all_tags)
    mask = tags == tag
    if not np.any(mask):
        warnings.warn(f"Nenhum tetra com tag {tag}.")
        return None

    sel = tets[mask]
    n = sel.shape[0]
    cells = np.hstack((np.full((n, 1), 4, np.int32), sel)).ravel()

    try:
        celltypes = np.full(n, pv.CellType.TETRA, np.uint8)
    except AttributeError:
        import vtk
        celltypes = np.full(n, vtk.VTK_TETRA, np.uint8)

    ug = pv.UnstructuredGrid(cells, celltypes, msh.points)
    surf = ug.extract_surface()
    try:
        surf = surf.clean(inplace=False).triangulate(inplace=False)
    except TypeError:
        surf = surf.clean().triangulate(inplace=False)

    surf.save(ply_path, binary=False)
    print(f"[PLY] salvo (ASCII): {ply_path}")
    return surf


# ============================================================
# MAIN com --mode roi / --mode greyzone
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extrai cicatriz a partir do .mat (ROI ou GreyZone) e gera STL para marcar fibrose."
    )
    parser.add_argument("matfile", help="Caminho para o .mat alinhado (saída do readMat)")
    parser.add_argument("--shiftx", default=None,
                        help="Caminho para endo_shifts_x.txt (usado em modo roi)")
    parser.add_argument("--shifty", default=None,
                        help="Caminho para endo_shifts_y.txt (usado em modo roi)")
    parser.add_argument("--output_path", required=True,
                        help="Pasta base do paciente (a mesma usada no execAll.py)")
    parser.add_argument("--patient_id", required=True,
                        help="Identificador do paciente")
    parser.add_argument(
        "--mode",
        choices=["roi", "greyzone"],
        default="roi",
        help="roi = usa setstruct.Roi; greyzone = usa Scar/GreyZone.map"
    )

    args = parser.parse_args()

    # Diretórios de saída (iguais para os dois modos)
    rois_dir = os.path.join(args.output_path, "rois_extruded")
    ply_dir  = os.path.join(args.output_path, "scarPly_raw")
    stl_dir  = os.path.join(args.output_path, "scarSTL_raw")

    if args.mode == "roi":
        print("===================================================")
        print("Modo ROI (setstruct.Roi)")
        print("===================================================")

        entries = readScar_roi(args.matfile)
        fatias = group_by_slice(entries)

        # Alinhamento em XY usando shifts do endo
        if args.shiftx is None or args.shifty is None:
            raise RuntimeError("Modo roi requer --shiftx e --shifty.")
        slices_dir = os.path.join(args.output_path, "slices")
        save_fatias_to_txt(fatias, args.shiftx, args.shifty, slices_dir)

        # Extrusão usando metadados do setstruct
        save_rois_extruded_to_txt(fatias, args.matfile, output_dir=rois_dir)

        # Geração de superfícies e STL
        generate_surfaces_and_stl(args.patient_id, rois_dir, ply_dir, stl_dir)

    else:
        print("===================================================")
        print("Modo GREYZONE (Scar/GreyZone.map → GZ + Core)")
        print("===================================================")

        lbl, res_x, res_y, dz = load_gz_map_and_metadata(args.matfile)
        print(f"Dimensões do mapa: {lbl.shape} (H, W, S)")
        print(f"ResolutionX={res_x} mm/pixel, ResolutionY={res_y} mm/pixel, dz={dz} mm")

        gz_slices   = extract_contours_all_slices(lbl, value=1, dx=res_x, dy=res_y, invert_y=False)
        core_slices = extract_contours_all_slices(lbl, value=2, dx=res_x, dy=res_y, invert_y=False)

        # Ajuste fixo em Z que você testou (1 unidade aqui)
        z_offset = 1.0
        print(f"[GZ] Usando z_offset fixo = {z_offset} (mesmas unidades de dz do Segment).")

        _ = save_region_extruded_txts(
            gz_slices,   dz, rois_dir, region_name="greyzone",
            num_layers=1, z_offset=z_offset
        )
        _ = save_region_extruded_txts(
            core_slices, dz, rois_dir, region_name="core",
            num_layers=1, z_offset=z_offset
        )

        generate_surfaces_and_stl(args.patient_id, rois_dir, ply_dir, stl_dir)

    print("===================================================")
    print("Fim do readScar.py.")
    print("===================================================")


if __name__ == "__main__":
    main()
