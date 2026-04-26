# MyoMesh

## Overview

**MyoMesh** is a pipeline for generating patient-specific heart meshes and converting them to a format suitable for electrophysiological simulations.

It integrates mesh processing, fiber assignment (via LDRB algorithm), and scar marking, producing final `.alg` files ready for use in simulators.

---

## Pre-Requisites

You do **not** need to install the individual libraries manually — the required dependencies are all included in the provided Conda environment.

**Dependencies included:**

- FEniCS 2019.1.0
- LDRB (via Conda)
- meshio
- h5py
- scipy
- CMake
- VTK 9.4.x (via Conda)
- Gmsh (binary included in `scripts/gmsh-2.13.1/`, no installation required)
- Other Python & C++ utilities (handled in the environment)

**Additional repository required:**

- [hexa-mesh-from-VTK_vtk9](https://github.com/FilipeNamorato/hexa-mesh-from-VTK_vtk9) — This will be cloned automatically during the configuration.

---

## Installation

1. Clone this repository:

   ```sh
   git clone https://github.com/FISIOCOMP-UFJF/MyoMesh
   cd MyoMesh
   ```
2. Create the Conda environment from the provided `.yml`:

   ```sh
   conda env create -f myomesh.yml
   ```
3. Activate the environment:

   ```sh
   conda activate myomesh
   ```

---

## Configuration

After activating the environment, run:

```sh
bash config.sh
```

or

```sh
sh config.sh
```

> **Shell compatibility:** `bash config.sh` works on most systems. If it fails (common on Fedora and other distributions where `bash` spawns a non-interactive subshell that does not inherit the Conda environment), use `sh config.sh` instead. The difference is in how each distribution's shell inherits the active Conda session.

This will:

- Clone and build the `hexa-mesh-from-VTK` project.
- Build the `convertPly2STL` executable used by the pipeline to convert surface meshes.

---

## Description parameters

- `-i`: Path to the file with heart meshes.
- `-r`: Discretization resolution in conversion to alg. Default value is 1000.
- `-dx`, `-dy`, and `-dz`: Refer to the discretization for the `.vtu`. Default value is 0.50.
- `--iterations`: Number of smoothing iterations applied to the scar/fibrosis surfaces (core and greyzone). Default value is 200.
- `--relaxation`: Relaxation factor controlling smoothing aggressiveness of the scar/fibrosis surfaces. Default value is 0.05.
- `--mesh_iterations`: Number of smoothing iterations applied to the cardiac surfaces (RVEndo, LVEpi, LVEndo). Default value is 40.
- `--mesh_relaxation`: Relaxation factor for cardiac surface smoothing. Default value is 0.02.
- `--cl_max`: Gmsh maximum element size (`CharacteristicLengthMax`). Default value is 2.0.
- `--cl_min`: Gmsh minimum element size (`CharacteristicLengthMin`). Default value is 1.0.
- `--no_invert_z`: Disable Z-axis flip. By default, Z is mirrored at the center of [z_min, z_max]. Pass this flag to keep the original Z orientation.
- `--alpha_endo_lv`: Fiber angle on the left ventricle (LV) endocardium. Default value is 30°.
- `--alpha_epi_lv`: Fiber angle on the left ventricle (LV) epicardium. Default value is -30°.
- `--beta_endo_lv`: Sheet angle on the left ventricle (LV) endocardium. Default value is 0°.
- `--beta_epi_lv`: Sheet angle on the left ventricle (LV) epicardium. Default value is 0°.
- `--alpha_endo_sept`: Fiber angle on the septum endocardium. Default value is 60°.
- `--alpha_epi_sept`: Fiber angle on the septum epicardium. Default value is -60°.
- `--beta_endo_sept`: Sheet angle on the septum endocardium. Default value is 0°.
- `--beta_epi_sept`: Sheet angle on the septum epicardium. Default value is 0°.
- `--alpha_endo_rv`: Fiber angle on the right ventricle (RV) endocardium. Default value is 80°.
- `--alpha_epi_rv`: Fiber angle on the right ventricle (RV) epicardium. Default value is -80°.
- `--beta_endo_rv`: Sheet angle on the right ventricle (RV) endocardium. Default value is 0°.
- `--beta_epi_rv`: Sheet angle on the right ventricle (RV) epicardium. Default value is 0°.

## Running the Pipeline

1. Activate the environment:

   ```sh
   conda activate myomesh
   ```
2. Run the main pipeline:

   ```sh
   python3 execAll.py -i path_to_patient_mat_file.mat
   ```

---

## Running Example

**Basic run (no fibrosis):**

```sh
python3 execAll.py -i ./Patient_1.mat
```

Runs the full pipeline. If the `.mat` file has no ROI or GreyZone data, scar marking is skipped and a clean biventricular mesh is generated.

---

**Patient with fibrosis/scar:**

```sh
python3 execAll.py -i ./Patient_1.mat
```

The pipeline automatically detects ROI or GreyZone data in the `.mat` file and runs the full scar marking workflow (core + greyzone segmentation, smoothing, and tagging).

---

**Custom mesh resolution and fiber angles:**

```sh
python3 execAll.py -i ./Patient_1.mat \
  -dx 0.50 -dy 0.50 -dz 0.50 -r 1000 \
  --cl_max 2.0 --cl_min 1.0 \
  --alpha_endo_lv 60 --alpha_epi_lv -60 \
  --alpha_endo_sept 60 --alpha_epi_sept -60 \
  --alpha_endo_rv 60 --alpha_epi_rv -60
```

---

**Keeping original Z-axis orientation (by default Z is mirrored):**

```sh
python3 execAll.py -i ./Patient_1.mat --no_invert_z
```

---

**Batch processing multiple patients:**

```sh
bash rodaPacientes.sh
```

or `sh rodaPacientes.sh` — see the Configuration section for notes on `bash` vs `sh` compatibility.

Runs `execAll.py` sequentially for each `.mat` file listed in the script.

---

**Full run with all explicit parameters:**

```sh
python3 execAll.py -i ./Patient_1.mat \
  -dx 0.50 -dy 0.50 -dz 0.50 -r 1000 \
  --iterations 200 --relaxation 0.05 \
  --mesh_iterations 200 --mesh_relaxation 0.02 \
  --cl_max 2.0 --cl_min 1.0 \
  --no_invert_z \
  --alpha_endo_lv 60 --alpha_epi_lv -60 --beta_endo_lv 0 --beta_epi_lv 0 \
  --alpha_endo_sept 60 --alpha_epi_sept -60 --beta_endo_sept 0 --beta_epi_sept 0 \
  --alpha_endo_rv 60 --alpha_epi_rv -60 --beta_endo_rv 0 --beta_epi_rv 0
```

---

## Notes

- The entire process is automated through `execAll.py`. You do not need to manually run intermediate scripts (saveMsh, makeSurface, readScar, etc.).
- The environment is fully self-contained.
- No additional system packages are required if you install via the provided Conda environment.
- Gmsh binary is already included and used by the pipeline.
- The `hexa-mesh-from-VTK` project will be cloned and compiled automatically during `config.sh`.

## How to cite

Namorato FL, Leme DMP, Soares TJ, Oliveira RS, Schmal TR, dos Santos RW, Campos JO. Development of a Three-Dimensional Computational Pipeline in Python for Personalized Heart Modeling. *Computing in Cardiology (CinC)*. 2025;52:1–4. https://doi.org/10.22489/CinC.2025.329
