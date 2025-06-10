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
   git clone https://github.com/FilipeNamorato/MyoMesh.git
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

This will:
- Clone and build the `hexa-mesh-from-VTK` project.
- Build the `convertPly2STL` executable used by the pipeline to convert surface meshes.
---

## Description parameters
- `-i`: Path to the file with heart meshes.
- `-o`: Name for the output file.
- `-r`: Discretization resolution in conversion to alg
- `-dx`, `-dy`, and `-dz`: refer to the discretization for the `.vtu`. Conventionally, we use the value of 0.5.
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

Basic run:
```sh
python3 execAll.py -i ./Patient_1.mat
```

Full run with explicit parameters:
```sh
python3 execAll.py -i ./Patient_1.mat -dx 0.5 -dy 0.5 -dz 0.5 -r 1000 --alpha_endo_lv 30 --alpha_epi_lv -30 --beta_endo_lv 0 --beta_epi_lv 0 --alpha_endo_sept 60 --alpha_epi_sept -60 --beta_endo_sept 0 --beta_epi_sept 0 --alpha_endo_rv 80 --alpha_epi_rv -80 --beta_endo_rv 0 --beta_epi_rv 0
```

---

## Notes

- The entire process is automated through `execAll.py`. You do not need to manually run intermediate scripts (saveMsh, makeSurface, readScar, etc.).
- The environment is fully self-contained.
- No additional system packages are required if you install via the provided Conda environment.
- Gmsh binary is already included and used by the pipeline.
- The `hexa-mesh-from-VTK` project will be cloned and compiled automatically during `config.sh`.
