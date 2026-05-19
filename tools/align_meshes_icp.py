"""
Aligns a source mesh (VTU/VTP/MSH) to a target mesh using ICP.

Usage:
    python3 align_meshes_icp.py --source <file> --target <file> --output <file.vtu>

The script:
  1. Samples point clouds from both surfaces
  2. Does a coarse alignment (center + long-axis orientation)
  3. Tries multiple flip combinations to find the best coarse init
  4. Runs ICP to refine
  5. Saves the transformed source mesh
"""

import argparse
import numpy as np
import pyvista as pv
import open3d as o3d
import meshio
import os


def load_surface_points(path, n_points=5000, scale=1.0):
    """Load mesh, extract surface, sample points uniformly."""
    ext = os.path.splitext(path)[1].lower()

    if ext in ('.vtu', '.vtp', '.vtk'):
        mesh = pv.read(path)
    elif ext == '.msh':
        mio = meshio.read(path)
        pts = mio.points
        cells_mio = [c for c in mio.cells if c.type == 'tetra']
        if cells_mio:
            tets = cells_mio[0].data
            n = tets.shape[0]
            cells_flat = np.hstack([np.full((n, 1), 4, dtype=int), tets]).ravel()
            celltypes = np.full(n, pv.CellType.TETRA, dtype=np.uint8)
            mesh = pv.UnstructuredGrid(cells_flat, celltypes, pts)
        else:
            mesh = pv.PolyData(pts)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    # Apply scale AFTER loading (e.g. scale=0.1 converts µm → mm)
    if scale != 1.0:
        mesh = mesh.copy()
        mesh.points = np.array(mesh.points) * scale

    surface = mesh.extract_surface()
    sampled = surface.sample_points_poisson_disk(n_points) if hasattr(surface, 'sample_points_poisson_disk') else surface.points
    if hasattr(sampled, 'points'):
        pts = np.array(sampled.points)
    else:
        pts = np.array(sampled)
    return pts, mesh


def to_o3d(pts):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.estimate_normals()
    return pc


def long_axis(pts):
    c = pts - pts.mean(0)
    vals, vecs = np.linalg.eigh(np.cov(c.T))
    return vecs[:, vals.argsort()[::-1]]


def coarse_align(src_pts, tgt_pts):
    """Align centroids and principal axes, trying flip combos, return best."""
    src_c = src_pts.mean(0)
    tgt_c = tgt_pts.mean(0)

    src_axes = long_axis(src_pts)
    tgt_axes = long_axis(tgt_pts)

    best_T = None
    best_dist = np.inf

    for s1 in [1, -1]:
        for s2 in [1, -1]:
            sa = src_axes.copy()
            sa[:, 0] *= s1
            sa[:, 1] *= s2
            sa[:, 2] = np.cross(sa[:, 0], sa[:, 1])

            R = tgt_axes @ sa.T

            if not np.isfinite(R).all():
                continue
            if np.linalg.det(R) < 0:
                continue

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tgt_c - R @ src_c

            src_aligned = apply_transform(src_pts, T)

            # score robusto: distância ao centróide do target
            dist = np.mean(np.linalg.norm(src_aligned - tgt_c, axis=1))

            if np.isfinite(dist) and dist < best_dist:
                best_dist = dist
                best_T = T

    if best_T is None:
        print("  WARNING: coarse alignment failed, falling back to centroid translation.")
        best_T = np.eye(4)
        best_T[:3, 3] = tgt_c - src_c
        best_dist = np.mean(np.linalg.norm(apply_transform(src_pts, best_T) - tgt_c, axis=1))

    print(f"  Best coarse candidate mean dist: {best_dist:.1f} mm")
    return best_T


def apply_transform(pts, T):
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (T @ h.T).T[:, :3]


def run_icp(src_pts, tgt_pts, init_T, max_iter=100, threshold=2.0, point_to_plane=True):
    src_pc = to_o3d(src_pts)
    tgt_pc = to_o3d(tgt_pts)

    if point_to_plane:
        result = o3d.pipelines.registration.registration_icp(
            src_pc, tgt_pc,
            max_correspondence_distance=threshold,
            init=init_T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
        )
    else:
        result = o3d.pipelines.registration.registration_icp(
            src_pc, tgt_pc,
            max_correspondence_distance=threshold,
            init=init_T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
        )

    return result.transformation, result.inlier_rmse, result.fitness


def transform_pyvista_mesh(mesh, T):
    pts = np.array(mesh.points)
    new_pts = apply_transform(pts, T)
    mesh_out = mesh.copy()
    mesh_out.points = new_pts
    return mesh_out


def main():
    parser = argparse.ArgumentParser(description="ICP mesh alignment")
    parser.add_argument("--source", required=True, help="Source mesh to be aligned (.vtu/.vtp/.msh)")
    parser.add_argument("--target", required=True, help="Target/reference mesh (.vtu/.vtp/.msh)")
    parser.add_argument("--output", required=True, help="Output aligned mesh (.vtu)")
    parser.add_argument("--n_points", type=int, default=5000, help="Points sampled per mesh for ICP")
    parser.add_argument("--threshold", type=float, default=5.0, help="ICP correspondence distance threshold (mm)")
    parser.add_argument("--no_coarse", action="store_true", help="Skip coarse alignment (use identity init)")
    parser.add_argument("--source_scale", type=float, default=0.1,
                        help="Scale factor for source mesh coordinates (default: 0.1, converts µm → mm)")
    args = parser.parse_args()

    print(f"Loading source: {args.source}  (scale={args.source_scale})")
    src_pts, src_mesh = load_surface_points(args.source, args.n_points, scale=args.source_scale)
    print(f"  {len(src_pts)} surface points")

    print(f"Loading target: {args.target}")
    tgt_pts, _ = load_surface_points(args.target, args.n_points, scale=1.0)
    print(f"  {len(tgt_pts)} surface points")

    if args.no_coarse:
        init_T = np.eye(4)
        print("Skipping coarse alignment (using identity)")
    else:
        print("Computing coarse alignment (trying all flip combinations)...")
        init_T = coarse_align(src_pts, tgt_pts)

    print(f"Running ICP (threshold={args.threshold}mm, point-to-plane)...")
    T_icp, rmse, fitness = run_icp(src_pts, tgt_pts, init_T,
                                    threshold=args.threshold, max_iter=200)
    print(f"  ICP done.  RMSE={rmse:.3f} mm  Fitness={fitness:.3f}")

    print("Applying transform to full mesh...")
    aligned_mesh = transform_pyvista_mesh(src_mesh, T_icp)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    aligned_mesh.save(args.output)
    print(f"Saved aligned mesh: {args.output}")

    R = T_icp[:3, :3]
    t = T_icp[:3, 3]
    print(f"\nFinal transform:")
    print(f"  Translation: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}] mm")
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"  Rotation angle: {angle:.1f}°")


if __name__ == "__main__":
    main()