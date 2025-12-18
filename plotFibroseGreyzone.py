import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage.measure import find_contours
import argparse

# ============================================================
# IMPORTA AS FUNÇÕES DO SEU readScar.py
# ============================================================
from src.mat2msh.readScar import (
    load_gz_map_and_metadata, 
    readScar_roi,             
    group_by_slice            
)

def load_heart_contours(mat_filename):
    print(f"[DEBUG] Lendo contornos cardíacos de: {mat_filename}")
    data = loadmat(mat_filename, struct_as_record=False, squeeze_me=True)
    ss = data['setstruct']

    def get_coords(field):
        val = getattr(ss, field, None)
        if val is None: return None
        if not isinstance(val, np.ndarray): val = np.array(val)
        if val.ndim == 1: val = val[:, np.newaxis]
        return val

    contours = {}
    mapping = [
        ('LV_Endo', 'EndoX', 'EndoY'),
        ('LV_Epi',  'EpiX',  'EpiY'),
        ('RV_Endo', 'RVEndoX', 'RVEndoY'),
        ('RV_Epi',  'RVEpiX',  'RVEpiY')
    ]

    for name, keyX, keyY in mapping:
        X = get_coords(keyX)
        Y = get_coords(keyY)
        if X is not None and Y is not None:
            contours[name] = (X, Y)
    
    return contours

# ============================================================
# DEBUG FINAL: TUDO EM COORDENADAS PYTHON (0-BASED)
# ============================================================
def debug_plot_roi_vs_greyzone(mat_path, output_dir="./debug_final_check", invert_y=False, show=False):
    # 1. Carrega Imagem
    lbl, res_x, res_y, dz = load_gz_map_and_metadata(mat_path)
    H, W, S = lbl.shape
    print(f"[DEBUG] Grid Python: H={H}, W={W}")

    # 2. Carrega Dados do Matlab
    heart_contours = load_heart_contours(mat_path)
    roi_entries = readScar_roi(mat_path)
    rois_by_slice = group_by_slice(roi_entries)

    os.makedirs(output_dir, exist_ok=True)
    
    for s in range(1, S + 1):
        idx = s - 1
        gz_slice = lbl[:, :, idx]
        has_content = np.any(gz_slice > 0)

        # Se quiser pular fatias vazias, descomente:
        # if not has_content: continue

        # Configura Plot
        fig, ax = plt.subplots(figsize=(10, 10)) # Aumentei um pouco a figura
        ax.set_title(f"Slice {s} (Idx {idx})\nRef: Python 0-based\n(Matlab -1.0 | Python Raw)")

        # --- 1. PIXELS (MÁSCARA GREYZONE/CORE) ---
        if has_content:
            base = (gz_slice > 0).astype(float)
            origin = 'lower' if invert_y else 'upper'
            ax.imshow(base, cmap="gray", origin=origin)

            # Contornos da Greyzone (Gerados no Python -> RAW)
            for val, color in [(1, 'yellow'), (2, 'red')]:
                mask = (gz_slice == val)
                if np.any(mask):
                    contours = find_contours(mask.astype(float), level=0.5)
                    for c in contours:
                        y_c, x_c = c[:, 0], c[:, 1]
                        
                        # AJUSTE: NENHUM (Python puro)
                        x_plot = x_c 
                        y_plot = y_c 
                        
                        if invert_y: y_plot = (H - 1) - y_c
                        
                        ax.plot(x_plot, y_plot, color=color, linewidth=1.5, alpha=0.9,
                                label=f"GZ Mask (Py)" if (s==1 and val==1) else "")

        # --- 2. LINHAS MATEMÁTICAS (LV/RV) ---
        # Como vêm do Matlab, aplicamos o mesmo -1.0 da ROI
        colors_math = {'LV_Endo': 'cyan', 'LV_Epi': 'blue', 'RV_Endo': 'magenta', 'RV_Epi': 'purple'}
        
        for name, (arrX, arrY) in heart_contours.items():
            if idx < arrX.shape[-1]:
                x_pts = arrX[:, ..., idx].flatten() 
                y_pts = arrY[:, ..., idx].flatten()
                valid = ~np.isnan(x_pts) & ~np.isnan(y_pts)
                x_pts = x_pts[valid]
                y_pts = y_pts[valid]

                if len(x_pts) == 0: continue

                # AJUSTE: SUBTRAIR 1.0 (Matlab -> Python)
                x_plot = x_pts - 1.0
                y_plot = y_pts - 1.0

                if invert_y: y_plot = (H - 1) - y_plot

                ax.plot(x_plot, y_plot, '--', color=colors_math.get(name, 'white'), 
                        linewidth=1, alpha=0.6, label=f"{name}" if s==1 else "")

        # --- 3. ROIS MANUAIS (VERDE) ---
        rois_desta_fatia = rois_by_slice.get(s, {})
        for roi_name, points in rois_desta_fatia.items():
            pts = np.array(points)
            if pts.size == 0: continue
            
            # AJUSTE: SUBTRAIR 1.0 (Matlab -> Python)
            roi_x_orig = pts[:, 0] - 1.0
            roi_y_orig = pts[:, 1] - 1.0
            
            x_roi_plot = np.r_[roi_x_orig, roi_x_orig[0]]
            y_roi_plot = np.r_[roi_y_orig, roi_y_orig[0]]

            if invert_y: y_roi_plot = (H - 1) - y_roi_plot
            
            ax.plot(x_roi_plot, y_roi_plot, '-', color='lime', 
                    linewidth=2.0, label=f"ROI {roi_name}" if s==1 else "")

        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        if not invert_y: ax.invert_yaxis()
        
        # Legenda externa para não poluir
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize='small')
        
        save_path = os.path.join(output_dir, f"slice_{s:03d}.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Salvo: {save_path}")

        if show:
            plt.show()
        
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("matfile")
    parser.add_argument("--out", default="./debug_final_check")
    parser.add_argument("--invert-y", action="store_true")
    parser.add_argument("--show", action="store_true", help="Mostra janelas interativas")
    
    args = parser.parse_args()
    
    debug_plot_roi_vs_greyzone(args.matfile, args.out, args.invert_y, args.show)