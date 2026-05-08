# MyoMesh — Full Documentation

## Table of Contents

- [Scientific Background](#scientific-background)
- [Pipeline Overview](#pipeline-overview)
- [Input Format](#input-format)
- [Output Format](#output-format)
- [Pre-Requisites](#pre-requisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Parameter Reference](#parameter-reference)
- [Examples](#examples)
- [Output Structure](#output-structure)
- [Batch Processing](#batch-processing)
- [Module Descriptions](#module-descriptions)
- [Notes](#notes)
- [References](#references)
- [How to cite](#how-to-cite)

---

## Scientific Background

Patient-specific cardiac modeling requires accurate three-dimensional representations of ventricular geometry, myocardial microstructure, and pathological tissue. MyoMesh automates the conversion of 2D MRI contour segmentations into volumetric finite element meshes, integrating:

- **Geometric reconstruction**: 2D contours (LV endocardium, LV epicardium, RV endocardium, RV epicardium) are extruded and triangulated into closed 3D surfaces, then meshed volumetrically using Gmsh with the Frontal-3D algorithm.
- **Slice alignment**: Slice-to-slice translational misalignment (common in breath-hold MRI acquisitions) is corrected by computing the barycenter of each contour per slice and fitting a linear regression, which is then subtracted as a shift.
- **Fiber orientation**: Myocardial fiber, sheet, and sheet-normal directions are assigned using the Laplace–Dirichlet Rule-Based (LDRB) method (Bayer et al., 2012), solving three Laplace boundary value problems with Dirichlet conditions on the endocardial, epicardial, and basal surfaces. Fiber helix and transverse angles are user-configurable per region (LV, septum, RV).
- **Scar and fibrosis delineation**: When the input contains scar annotations (manual ROI or GreyZone maps from LGE-MRI), the pipeline segments and volumetrically tags myocardial regions: healthy tissue (tag 0), dense scar/core (tag 2), and border zone/grey zone (tag 3). Scar surfaces are extracted, smoothed, and used to mark tetrahedra whose centroids fall within the scar volume.
- **Hexahedral output**: The tetrahedral mesh is resampled onto a regular hexahedral grid (`.alg`) for use in simulators requiring structured meshes, preserving fiber vectors and tissue tags per voxel.

---

## Pipeline Overview

```
Patient_X.mat  (2D MRI segmentations)
      │
      ▼
[1] readMat.py
      Reads LV/RV contours, computes slice barycenters,
      fits linear regression to correct translational misalignment,
      saves aligned .mat and shift files.
      │
      ▼
[2] saveMsh.py
      Projects 2D contours to 3D using slice thickness and gap.
      Applies alignment shifts. Optionally mirrors Z-axis around
      the center of [z_min, z_max] for anatomical orientation.
      Output: TXT files with (X, Y, Z) point clouds per surface.
      │
      ▼
[3] makeSurface.py
      Reads TXT point clouds and builds closed triangulated 3D surfaces.
      Resamples contours to uniform point count, aligns ring phases
      between consecutive slices, connects rings with triangles,
      and closes the apex end.
      Output: PLY files.
      │
      ▼
[4] PlyToStl (convertPly2STL)
      Applies Laplacian smoothing and converts PLY → STL.
      Separate smoothing parameters for cardiac surfaces and fibrosis surfaces.
      Output: STL files.
      │
      ▼
[5] Gmsh (biv_mesh.geo)
      Generates a tetrahedral biventricular mesh from 4 STL surfaces
      (LVEndo, RVEndo, RVEpi, LVEpi) using the Frontal-3D algorithm.
      Tags physical surfaces: BASE=10, RV=20, LV=30, EPI=40.
      Applies Lloyd smoothing for mesh quality.
      Output: Patient_X.msh
      │
      ├──[if fibrosis detected]──────────────────────────────────────────┐
      │                                                                   │
      │  [6] readScar.py                                                  │
      │        Extracts scar contours from .mat (ROI or GreyZone.map).   │
      │        Transposes GreyZone matrix (X↔Y correction).              │
      │        Extrudes 2D contours into 3D volumes.                     │
      │        Output: STL surfaces for core and grey zone.              │
      │                                                                   │
      │  [7] markFibroseFromMsh.py                                        │
      │        For each tetrahedron, tests if centroid lies inside        │
      │        the scar STL. Tags elements: core=2, grey zone=3.         │
      │        Applies Laplacian smoothing to scar surfaces.             │
      │        Output: Patient_X_marked_smooth.msh                       │
      └───────────────────────────────────────────────────────────────────┘
      │
      ▼
[8] msh2alg.py
      Mirrors mesh in X (anatomical convention).
      Converts MSH → XML using dolfin-convert (FEniCS).
      │
      ▼
[9] generate_fiber_3D_biv.py
      Reads XML mesh and boundary markers.
      Solves 3 Laplace problems (φ_lv, φ_rv, φ_epi) with GMRES + AMG.
      Calls ldrb.dolfin_ldrb() to compute fiber (f_0), sheet (s_0),
      and sheet-normal (n_0) vectors with user-defined angles.
      Tags tissue regions (healthy=0, core=1, grey zone=2).
      Converts output to VTU for visualization.
      Output: Patient_X.xdmf, Patient_X.vtu, Patient_X.pts/.elem/.lon
      │
      ▼
[10] HexaMeshFromVTK
      Resamples the tetrahedral VTU onto a regular hexahedral grid
      with configurable voxel size (dx, dy, dz) and resolution.
      Transfers fiber vectors and tissue tags to voxels.
      Output: Patient_X.alg
      │
      ▼
Patient_X.alg         — hexahedral mesh with fiber orientations and tissue tags
Patient_X.vtu         — tetrahedral mesh for visualization (ParaView)
Patient_X.pts/.elem/.lon — OpenCARP format
```

---

## Input Format

MyoMesh expects a `.mat` file (MATLAB format) containing cardiac segmentations from cine or late-gadolinium-enhancement (LGE) MRI. The file must contain a `setstruct` variable with the following fields:

| Field                            | Description                                           |
| -------------------------------- | ----------------------------------------------------- |
| `EndoX`, `EndoY`             | LV endocardial contour coordinates (points × slices) |
| `EpiX`, `EpiY`               | LV epicardial contour coordinates                     |
| `RVEndoX`, `RVEndoY`         | RV endocardial contour coordinates                    |
| `RVEpiX`, `RVEpiY`           | RV epicardial contour coordinates                     |
| `SliceThickness`               | Slice thickness in mm (default: 8 mm if absent)       |
| `SliceGap`                     | Inter-slice gap in mm (default: 0.64 mm if absent)    |
| `ResolutionX`, `ResolutionY` | Pixel spacing in mm                                   |
| `Roi` *(optional)*           | Manual ROI scar annotations                           |
| `GreyZone.map` *(optional)*  | 3D label matrix: 0 = healthy, 1 = grey zone, 2 = core |

The pipeline automatically detects the presence of `Roi` or `GreyZone` data and activates the fibrosis workflow.

---

## Output Format

| File               | Format                | Description                                                  |
| ------------------ | --------------------- | ------------------------------------------------------------ |
| `Patient_X.alg`  | ALG hexahedral        | Primary simulation mesh with fiber vectors and tissue tags   |
| `Patient_X.vtu`  | VTK Unstructured Grid | Tetrahedral mesh with f_0, s_0, n_0 fields for visualization |
| `Patient_X.xdmf` | XDMF/HDF5             | FEniCS output with fiber/sheet/normal fields                 |
| `Patient_X.pts`  | OpenCARP              | Node coordinates                                             |
| `Patient_X.elem` | OpenCARP              | Element connectivity and tissue tags                         |
| `Patient_X.lon`  | OpenCARP              | Fiber and sheet vectors per element                          |
| `pipeline.log`   | Log                   | Full timestamped execution log                               |

**Tissue tags:**

| Tag | Tissue                  |
| --- | ----------------------- |
| 0   | Healthy myocardium      |
| 2   | Dense scar (core)       |
| 3   | Border zone (grey zone) |

**Surface boundary tags (Gmsh / LDRB boundary conditions):**

| Tag | Surface        |
| --- | -------------- |
| 10  | Base           |
| 20  | RV endocardium |
| 30  | LV endocardium |
| 40  | Epicardium     |

---

## Pre-Requisites

All dependencies are provided in the Conda environment. No manual installation is required.

**Python libraries:**

- NumPy, SciPy, h5py, meshio, pyvista, scikit-image
- FEniCS 2019.1.0 — PDE solver for Laplace boundary value problems
- ldrb — Laplace–Dirichlet Rule-Based fiber assignment
- VTK 9.4.x

**Included binaries:**

- Gmsh 2.13.1 — tetrahedral mesh generation (`scripts/gmsh-2.13.1/`)
- PlyToStl — Laplacian surface smoothing and PLY→STL conversion (`convertPly2STL/`)

**External repository (cloned automatically by `config.sh`):**

- [hexa-mesh-from-VTK_vtk9](https://github.com/FilipeNamorato/hexa-mesh-from-VTK_vtk9) — tetrahedral to hexahedral mesh resampling

---

## Installation

```sh
git clone https://github.com/FISIOCOMP-UFJF/MyoMesh
cd MyoMesh
conda env create -f myomesh.yml
conda activate myomesh
```

---

## Configuration

Run once after activating the environment:

```sh
bash config.sh
```

or

```sh
sh config.sh
```

> **Shell compatibility:** `bash config.sh` works on most systems. If it fails (common on Fedora and other distributions where `bash` spawns a non-interactive subshell that does not inherit the Conda environment), use `sh config.sh` instead. The difference is in how each distribution's shell inherits the active Conda session.

This clones and compiles `hexa-mesh-from-VTK_vtk9` and builds `convertPly2STL`.

---

## Running the Pipeline

```sh
conda activate myomesh
python3 execAll.py -i path/to/Patient_X.mat
```

All steps are orchestrated automatically. Outputs are saved to `output/YYYYMMDD_HHMM/PatientID/`.

---

## Parameter Reference

### Input

| Parameter            | Description                     | Default        |
| -------------------- | ------------------------------- | -------------- |
| `-i, --input_file` | Path to the `.mat` input file | *(required)* |

### Mesh Resolution

| Parameter            | Description                                                    | Default  |
| -------------------- | -------------------------------------------------------------- | -------- |
| `-r, --resolution` | Discretization resolution for hexahedral grid                  | `1000` |
| `--dx`             | Voxel size in X (mm)                                           | `0.50` |
| `--dy`             | Voxel size in Y (mm)                                           | `0.50` |
| `--dz`             | Voxel size in Z (mm)                                           | `0.50` |
| `--cl_max`         | Gmsh `CharacteristicLengthMax`: max tetrahedral element size | `2.0`  |
| `--cl_min`         | Gmsh `CharacteristicLengthMin`: min tetrahedral element size | `1.0`  |

### Surface Smoothing

| Parameter             | Description                                          | Default  |
| --------------------- | ---------------------------------------------------- | -------- |
| `--mesh_relaxation` | Laplacian relaxation factor for cardiac surfaces     | `0.02` |
| `--mesh_iterations` | Laplacian smoothing iterations for cardiac surfaces  | `40`   |
| `--relaxation`      | Laplacian relaxation factor for fibrosis surfaces    | `0.05` |
| `--iterations`      | Laplacian smoothing iterations for fibrosis surfaces | `200`  |

### Geometry

| Parameter         | Description                                                                                                                                                                           | Default  |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `--no_invert_z` | Disable Z-axis mirroring. By default Z is reflected around the center of `[z_min, z_max]` to match anatomical orientation. Pass this flag to preserve the original MRI slice order. | disabled |

### Fiber Angles (LDRB)

Angles follow the convention of Bayer et al. (2012): the helix angle (α) is measured from the circumferential direction in the tangent plane, and the transverse angle (β) is the elevation out of that plane. Positive α = counterclockwise when viewed from the base.

| Parameter             | Description                            | Default   |
| --------------------- | -------------------------------------- | --------- |
| `--alpha_endo_lv`   | Helix angle at LV endocardium          | `30°`  |
| `--alpha_epi_lv`    | Helix angle at LV epicardium           | `-30°` |
| `--beta_endo_lv`    | Transverse angle at LV endocardium     | `0°`   |
| `--beta_epi_lv`     | Transverse angle at LV epicardium      | `0°`   |
| `--alpha_endo_sept` | Helix angle at septum endocardium      | `60°`  |
| `--alpha_epi_sept`  | Helix angle at septum epicardium       | `-60°` |
| `--beta_endo_sept`  | Transverse angle at septum endocardium | `0°`   |
| `--beta_epi_sept`   | Transverse angle at septum epicardium  | `0°`   |
| `--alpha_endo_rv`   | Helix angle at RV endocardium          | `80°`  |
| `--alpha_epi_rv`    | Helix angle at RV epicardium           | `-80°` |
| `--beta_endo_rv`    | Transverse angle at RV endocardium     | `0°`   |
| `--beta_epi_rv`     | Transverse angle at RV epicardium      | `0°`   |

---

## Examples

**Basic run (no fibrosis):**

```sh
python3 execAll.py -i ./Patient_1.mat
```

**Patient with fibrosis — automatic detection:**

```sh
python3 execAll.py -i ./Patient_1.mat
```

If `Roi` or `GreyZone` fields are present in the `.mat` file, the full scar workflow runs automatically (contour extraction → STL generation → smoothing → volumetric tagging: core=2, grey zone=3).

**Custom mesh density:**

```sh
python3 execAll.py -i ./Patient_1.mat --cl_max 1.5 --cl_min 0.8
```

**Custom fiber angles:**

```sh
python3 execAll.py -i ./Patient_1.mat \
  --alpha_endo_lv 70 --alpha_epi_lv -70 \
  --alpha_endo_rv 50 --alpha_epi_rv -50
```

**Preserve original MRI Z-axis orientation:**

```sh
python3 execAll.py -i ./Patient_1.mat --no_invert_z
```

**Full run with all parameters explicit:**

```sh
python3 execAll.py -i ./Patient_1.mat \
  -dx 0.50 -dy 0.50 -dz 0.50 -r 1000 \
  --cl_max 2.0 --cl_min 1.0 \
  --mesh_iterations 200 --mesh_relaxation 0.02 \
  --iterations 200 --relaxation 0.05 \
  --alpha_endo_lv 60 --alpha_epi_lv -60 --beta_endo_lv 0 --beta_epi_lv 0 \
  --alpha_endo_sept 60 --alpha_epi_sept -60 --beta_endo_sept 0 --beta_epi_sept 0 \
  --alpha_endo_rv 60 --alpha_epi_rv -60 --beta_endo_rv 0 --beta_epi_rv 0
```

---

## Output Structure

```

output/
└── YYYYMMDD_HHMM/
    └── PatientID/
        ├── pipeline.log                        # Full timestamped execution log
        ├── PatientID.mat                       # Aligned segmentation file
        ├── endo_shifts_x.txt                   # Slice alignment shifts (removed after run)
        ├── endo_shifts_y.txt
        ├── patientTxt/                         # 3D point clouds per surface
        │   ├── PatientID-LVEndo.txt
        │   ├── PatientID-LVEpi.txt
        │   ├── PatientID-RVEndo.txt
        │   └── PatientID-RVEpi.txt
        ├── patientPly/                         # Triangulated surfaces (PLY)
        ├── patientSTL/                         # Smoothed surfaces (STL)
        │   ├── PatientID-LVEndo.stl
        │   ├── PatientID-LVEpi.stl
        │   ├── PatientID-RVEndo.stl
        │   ├── PatientID-RVEpi.stl
        │   └── PatientID-BASE.stl
        ├── patientMsh/                         # Gmsh tetrahedral meshes
        │   ├── PatientID.msh                   # Full biventricular mesh
        │   ├── PatientID_lv.msh                # LV-only mesh (if fibrosis)
        │   └── PatientID_marked_smooth.msh     # Mesh with fibrosis tags
        ├── scarPly/                            # Fibrosis surfaces (PLY)
        ├── scarSTL/                            # Fibrosis surfaces (STL)
        ├── logs/                               # Gmsh and HexaMesh execution logs
        │   ├── PatientID_gmsh.log
        │   ├── PatientID_gmsh_lv.log
        │   └── PatientID_hexa.log
        ├── fenicsFiles/                        # FEniCS solver output
        │   ├── PatientID.xml
        │   ├── PatientID_facet_region.xml
        │   ├── PatientID_physical_region.xml
        │   ├── PatientID.h5
	|   ├── PatientID.xdmf
        │   └── PatientID.vtu
        ├── carpFiles/                          # OpenCARP format
        │   ├── PatientID.pts
        │   ├── PatientID.elem
        │   ├── PatientID.fib
        │   └── PatientID.lon
	|
        └── algFiles/                           # Hexahedral mesh (primary output)
            └── PatientID.alg
```

---

## Batch Processing

Edit `rodaPacientes.sh` to list the patient `.mat` file paths, then:

```sh
bash rodaPacientes.sh
```

> See the [Configuration](#configuration) section for notes on `bash` vs `sh` compatibility.

Each patient is processed sequentially. Stdout and stderr are captured per patient. A `summary.log` is written with the status of each execution.

---

## Module Descriptions

### `execAll.py`

Main orchestrator. Parses CLI arguments, creates the output directory structure, sets up logging, and runs all pipeline steps in order via subprocess calls and direct function calls.

### `src/mat2msh/readMat.py`

Reads the `.mat` file and extracts contour coordinates for all four cardiac surfaces. Computes per-slice barycenters and fits a linear regression to correct translational misalignment between slices. Saves the aligned `.mat` and shift `.txt` files used in subsequent steps.

### `src/mat2msh/saveMsh.py`

Converts the aligned 2D contours to 3D point clouds by assigning Z coordinates from slice thickness and gap. Applies the alignment shifts from `readMat.py` and optionally mirrors the Z-axis around the center of the Z range. Outputs one `.txt` file per cardiac surface.

### `src/mat2msh/makeSurface.py`

Reads a `.txt` point cloud and builds a closed triangulated 3D surface. Resamples each contour ring to a uniform number of points, aligns ring phases between consecutive slices to minimize edge crossings, connects rings with triangles, and closes the apex with a cone cap. Outputs a `.ply` file.

### `src/mat2msh/readScar.py`

Extracts scar/fibrosis regions from the `.mat` file. Supports two modes:

- **ROI mode**: reads manual contour annotations from `setstruct.Roi`.
- **GreyZone mode**: reads a 3D label matrix (`GreyZone.map`) with labels 0 (healthy), 1 (grey zone), 2 (core), detects contours per slice using scikit-image, and extrudes them into 3D volumes.

Outputs STL surfaces for each fibrosis class.

### `src/mat2msh/markFibroseFromMsh.py`

Marks fibrotic tetrahedra in the volumetric mesh. For each element, tests whether its centroid lies inside a fibrosis STL surface (ray casting). Assigns tags: core=2, grey zone=3. Outputs a tagged `.msh` file.

### `src/msh2alg/generate_fiber_3D_biv.py`

Core scientific module. Reads the FEniCS XML mesh, identifies boundary surfaces by tag, solves three Laplace problems (for the LV, RV, and epicardial scalar fields) using GMRES with AMG preconditioning, then calls `ldrb.dolfin_ldrb()` to compute rule-based fiber, sheet, and sheet-normal vectors with the user-specified angles. Saves results as XDMF/HDF5 and converts to VTU. Also provides utilities: `extract_base_stl()`, `mirror_msh_x()`, `msh_tag_to_ply()`.

### `src/msh2alg/msh2alg.py`

Orchestrates the tetrahedral-to-hexahedral conversion: mirrors the mesh, converts from `.msh` to `.xml` with `dolfin-convert`, calls `generate_fiber_3D_biv.request_functions()` to assign fibers, then runs `HexaMeshFromVTK` to resample onto the hexahedral grid.

### `src/msh2alg/msh2carp.py`

Converts the tagged tetrahedral mesh to OpenCARP format (`.pts`, `.elem`, `.lon`), writing node coordinates, element connectivity with tissue tags, and fiber/sheet vectors per element.

### `scripts/biv_mesh.geo`

Gmsh geometry script defining the biventricular mesh topology. Specifies compound surfaces for LV, RV, and epicardium; assigns physical surface tags (BASE=10, RV=20, LV=30, EPI=40); creates the wall volume; and enables Lloyd smoothing.

---

## Notes

- Individual scripts (`saveMsh.py`, `makeSurface.py`, `readScar.py`, etc.) can be run standalone for debugging, each accepting their own CLI arguments.
- The Gmsh binary (v2.13.1) is included in `scripts/gmsh-2.13.1/` — no separate installation needed.
- Laplace problems are solved with FEniCS 2019.1.0 (GMRES + AMG). Mesh size and element quality directly affect solver convergence.
- Reducing `--cl_max` produces a finer tetrahedral mesh and more accurate fiber gradients, at higher computational cost.
- The `--no_invert_z` flag is patient-specific: some MRI acquisitions store slices base-to-apex, others apex-to-base. Check the output geometry in ParaView if the mesh appears inverted.
- **Surface smoothing and self-intersections:** Low values of `--mesh_iterations` or `--mesh_relaxation` may produce self-intersecting STL surfaces, especially near the apex and RV insertion points, causing Gmsh to fail or generate a degenerate mesh. Increasing `--mesh_iterations` to 200 typically resolves this. However, excessive smoothing on patients with fibrosis may displace cardiac surface boundaries away from the scar region, compromising volumetric fibrosis marking accuracy. A value between 40 (default) and 200 should be chosen based on visual inspection of the output geometry.

---

## References

- Bayer, J. D., Blake, R. C., Plank, G., & Trayanova, N. A. (2012). *A novel rule-based algorithm for assigning myocardial fiber orientation to computational heart models*. Annals of Biomedical Engineering, 40(10), 2243–2254. https://doi.org/10.1007/s10439-012-0593-5
- Finsberg, H., et al. (n.d.). *ldrb: A Python library for assigning myocardial fiber orientations*. GitHub repository. https://github.com/finsberg/ldrbGeuzaine, C., & Remacle, J.-F. (2009). *Gmsh: A 3-D finite element mesh generator with built-in pre- and post-processing facilities*. International Journal for Numerical Methods in Engineering, 79(11), 1309–1331.
- Logg, A., Mardal, K.-A., & Wells, G. N. (Eds.). (2012). *Automated solution of differential equations by the finite element method*. Springer. https://doi.org/10.1007/978-3-642-23099-8

## How to Cite

> Namorato, F. L., Leme, D. M. P., Soares, T. J., Oliveira, R. S., Schmal, T. R., dos Santos, R. W., & Campos, J. O. (2025). *Development of a three-dimensional computational pipeline in Python for personalized heart modeling*. Computing in Cardiology (CinC), 52, 1–4. https://doi.org/10.22489/CinC.2025.329
