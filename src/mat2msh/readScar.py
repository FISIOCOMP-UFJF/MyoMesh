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
    """
    print(f"[ROI] Reading ROIs from: {mat_filename}")
    data = loadmat(mat_filename, struct_as_record=False, squeeze_me=True)

    roi_cell = data['setstruct'].Roi
    entries = []

    for idx, roi in enumerate(roi_cell):
        name = getattr(roi, 'Name', f'ROI-{idx+1}')
        if isinstance(name, np.ndarray):
            name = name[0] if name.size > 0 else f'ROI-{idx+1}'
        name = str(name)

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
    fatias = defaultdict(lambda: defaultdict(list))
    for e in entries:
        fatias[e.z][e.name].extend(e.points)
    return fatias


def save_fatias_to_txt(fatias, shifts_x_file, shifts_y_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    shifts_x = np.loadtxt(shifts_x_file)
    shifts_y = np.loadtxt(shifts_y_file)

    for z, roi_map in sorted(fatias.items()):
        # Se z for 1-based (Matlab), o índice do array é z-1
        idx_shift = z - 1
        sx = shifts_x[idx_shift] if 0 <= idx_shift < len(shifts_x) else 0
        sy = shifts_y[idx_shift] if 0 <= idx_shift < len(shifts_y) else 0
        
        fname = os.path.join(output_dir, f"slice_{z}.txt")
        with open(fname, 'w') as f:
            for pts in roi_map.values():
                for x, y in pts:
                    f.write(f"{x - sx} {y - sy} {z}\n")
        print(f"[ROI] Saved aligned slice {z} to {fname}")


def save_rois_extruded_to_txt(fatias, mat_filename, output_dir, num_layers=1):
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
# GreyZone.map -> GZ/Core
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

    # =========================================================
    # Transpose (Troca X <-> Y)
    # =========================================================
    print("[INFO] Aplicando Transpose na matriz GreyZone (Correção Geométrica)")
    lbl = np.transpose(lbl, (1, 0, 2))

    try:
        slice_thickness = float(ss["SliceThickness"])
        slice_gap       = float(ss["SliceGap"])
        
        # Como transpomos a imagem, trocamos as resoluções também
        res_x_raw    = float(ss["ResolutionX"])
        res_y_raw    = float(ss["ResolutionY"])
        resolution_x = res_y_raw
        resolution_y = res_x_raw
        
    except Exception as e:
        raise RuntimeError(
            "Problema ao ler SliceThickness/SliceGap/ResolutionX/ResolutionY em setstruct."
        ) from e

    dz = slice_thickness + slice_gap
    return lbl, resolution_x, resolution_y, dz


def extract_contours_all_slices(
    lbl,
    value,
    dx=1.0,
    dy=1.0,
    shifts_x=None,
    shifts_y=None,
    invert_y=False,
):
    H, W, S = lbl.shape

    sx_arr = np.asarray(shifts_x) if shifts_x is not None else None
    sy_arr = np.asarray(shifts_y) if shifts_y is not None else None

    all_slices = []
    for s in range(S):
        mask = (lbl[:, :, s] == value).astype(float)
        cs = find_contours(mask, level=0.5)
        polys = []

        sx = sx_arr[s] if sx_arr is not None and 0 <= s < len(sx_arr) else 0.0
        sy = sy_arr[s] if sy_arr is not None and 0 <= s < len(sy_arr) else 0.0
        #sx = 0
        #sy = 0
        
        for c in cs:
            # Mantem y=c[:,0] e x=c[:,1]
            # Isso garante que a geometria não fique espelhada.
            y_raw, x_raw = c[:, 0], c[:, 1]

            # ============================================================
            # Ajuste
            # ============================================================
            y = y_raw + 1
            x = x_raw + 1
            # ============================================================
            
            x_corr = x - sx
            y_corr = y - sy

            X = x_corr * dx
            Y = (H - 1 - y_corr) * dy if invert_y else y_corr * dy

            polys.append(np.column_stack([X, Y]))
        all_slices.append(polys)
    return all_slices

def align_fibrosis_txts_to_lvendo(output_path, patient_id, rois_dir):
    """
    Alinha todos os TXT de fibrose (rois_extruded/roi_*.txt) ao LVEndo.
    (Essa função é mantida, mas não será chamada se decidirmos usar --no-align)
    """

    lv_txt = os.path.join(output_path, "patientTxt", f"{patient_id}-LVEndo.txt")
    if not os.path.exists(lv_txt):
        print(f"[ALIGN] LVEndo txt '{lv_txt}' não encontrado. Pulando alinhamento XY.")
        return

    try:
        lv_pts = np.loadtxt(lv_txt)
        if lv_pts.ndim == 1:
            lv_pts = lv_pts[None, :]
        c_lv = lv_pts[:, :2].mean(axis=0)
    except Exception as e:
        print(f"[ALIGN] Erro ao ler LVEndo '{lv_txt}': {e}")
        return

    fib_files = sorted(glob.glob(os.path.join(rois_dir, "roi_*.txt")))
    if not fib_files:
        print(f"[ALIGN] Nenhum TXT de fibrose encontrado em '{rois_dir}'.")
        return

    fib_pts_list = []
    for fpath in fib_files:
        try:
            arr = np.loadtxt(fpath)
            if arr.ndim == 1:
                arr = arr[None, :]
            fib_pts_list.append(arr[:, :2])
        except Exception as e:
            print(f"[ALIGN] Aviso: não consegui ler '{fpath}': {e}")

    if not fib_pts_list:
        print("[ALIGN] Não foi possível montar pontos de fibrose para alinhamento.")
        return

    fib_pts = np.vstack(fib_pts_list)
    c_fib = fib_pts.mean(axis=0)

    delta_xy = c_lv - c_fib
    print(f"[ALIGN] Centróide LVEndo XY = {c_lv}")
    print(f"[ALIGN] Centróide fibrose XY = {c_fib}")
    print(f"[ALIGN] delta_xy (LVEndo - fibrose) = {delta_xy}")

    for fpath in fib_files:
        try:
            arr = np.loadtxt(fpath)
            if arr.ndim == 1:
                arr = arr[None, :]

            arr[:, 0] += delta_xy[0]
            arr[:, 1] += delta_xy[1]

            np.savetxt(fpath, arr, fmt="%.6f")
            print(f"[ALIGN] delta_xy aplicado em '{fpath}'")
        except Exception as e:
            print(f"[ALIGN] Erro ao aplicar delta_xy em '{fpath}': {e}")

def save_region_extruded_txts(polys_per_slice, dz, out_dir, region_name,
                              num_layers=1, z_offset=0.0):
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


def generate_surfaces_and_stl(patient_id, rois_dir, ply_dir, stl_dir):
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
# msh_tag_to_ply
# ============================================================

def msh_tag_to_ply(msh_path, tag=2, ply_path="fibrose_surface.ply"):
    """
    Lê um .msh com tetraedros e tags gmsh:physical,
    extrai os elementos com a tag especificada e
    salva a superfície correspondente em um .ply ASCII.
    """
    msh_path = Path(msh_path)
    ply_path = Path(ply_path)

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
    mask = (tags == tag)
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

def main():
    parser = argparse.ArgumentParser(
        description="Extrai cicatriz a partir do .mat (ROI ou GreyZone) e gera STL para marcar fibrose."
    )
    parser.add_argument("matfile", help="Caminho para o .mat alinhado (saída do readMat)")
    parser.add_argument("--shiftx", default=None,
                        help="Caminho para endo_shifts_x.txt (usado em modo roi/greyzone)")
    parser.add_argument("--shifty", default=None,
                        help="Caminho para endo_shifts_y.txt (usado em modo roi/greyzone)")
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

    rois_dir = os.path.join(args.output_path, "rois_extruded")
    ply_dir  = os.path.join(args.output_path, "scarPly_raw")
    stl_dir  = os.path.join(args.output_path, "scarSTL_raw")

    if args.mode == "roi":
        print("===================================================")
        print("Modo ROI (setstruct.Roi)")
        print("===================================================")

        entries = readScar_roi(args.matfile)
        fatias = group_by_slice(entries)

        if args.shiftx is None or args.shifty is None:
            raise RuntimeError("Modo roi requer --shiftx e --shifty.")
        slices_dir = os.path.join(args.output_path, "slices")
        save_fatias_to_txt(fatias, args.shiftx, args.shifty, slices_dir)

        save_rois_extruded_to_txt(fatias, args.matfile, output_dir=rois_dir)
        generate_surfaces_and_stl(args.patient_id, rois_dir, ply_dir, stl_dir)

    else:
        print("===================================================")
        print("Modo GREYZONE (Scar/GreyZone.map → GZ + Core)")
        print("===================================================")

        lbl, res_x, res_y, dz = load_gz_map_and_metadata(args.matfile)
        print(f"Dimensões do mapa: {lbl.shape} (H, W, S)")
        print(f"ResolutionX={res_x} mm/pixel, ResolutionY={res_y} mm/pixel, dz={dz} mm")

        shifts_x = np.loadtxt(args.shiftx) if args.shiftx is not None else None
        shifts_y = np.loadtxt(args.shifty) if args.shifty is not None else None

        gz_slices = extract_contours_all_slices(
            lbl, value=1,
            dx=res_x, dy=res_y,
            shifts_x=shifts_x, shifts_y=shifts_y,
            invert_y=False
        )
        core_slices = extract_contours_all_slices(
            lbl, value=2,
            dx=res_x, dy=res_y,
            shifts_x=shifts_x, shifts_y=shifts_y,
            invert_y=False
        )

        z_offset = 1.0 * dz
        print(f"[GZ] Usando z_offset = {z_offset} com correção de shifts por fatia.")

        _ = save_region_extruded_txts(gz_slices, dz, rois_dir, "greyzone", 1, z_offset)
        _ = save_region_extruded_txts(core_slices, dz, rois_dir, "core", 1, z_offset)

        generate_surfaces_and_stl(args.patient_id, rois_dir, ply_dir, stl_dir)

    print("===================================================")
    print("Fim do readScar.py.")
    print("===================================================")


if __name__ == "__main__":
    main()