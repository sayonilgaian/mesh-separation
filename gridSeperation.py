import copy
import os
import numpy as np
import open3d as o3d
from pathlib import Path

def segment_mesh_by_uniform_grid(input_mesh_path: str, output_dir: str, units: int = 2):
    mesh = o3d.io.read_triangle_mesh(input_mesh_path)
    if not mesh.has_triangles():
        raise ValueError("Mesh has no triangles.")
    mesh.compute_vertex_normals()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bbox = mesh.get_axis_aligned_bounding_box()
    min_b, max_b = bbox.min_bound, bbox.max_bound
    step = (max_b - min_b) / units

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    total_cells = units ** 3
    saved = 0

    for i in range(units):
      for j in range(units):
        for k in range(units):
            cube_min = min_b + np.array([i,j,k]) * step
            cube_max = cube_min + step
            sel = []
            for idx, tri in enumerate(triangles):
                centroid = vertices[tri].mean(axis=0)
                if np.all(centroid >= cube_min) and np.all(centroid <= cube_max):
                    sel.append(idx)
            if not sel:
                continue

            mesh_copy = copy.deepcopy(mesh)
            remove_idx = list(set(range(len(triangles))) - set(sel))
            mesh_copy.remove_triangles_by_index(remove_idx)
            mesh_copy.remove_unreferenced_vertices()
            mesh_copy.remove_duplicated_vertices()
            mesh_copy.remove_degenerate_triangles()

            if not mesh_copy.has_triangles():
                continue

            fname = f"part_{i}_{j}_{k}.glb"
            out = output_path / fname
            good = o3d.io.write_triangle_mesh(str(out), mesh_copy,
                                              write_ascii=False,
                                              write_triangle_uvs=True)
            if good:
                print(f"✅ Saved {fname} ({len(mesh_copy.triangles)} triangles)")
                saved += 1
            else:
                print(f"❌ Failed {fname}")

    print(f"\n🎉 Done: {saved}/{total_cells} cells saved to {output_dir}")

if __name__ == "__main__":
    segment_mesh_by_uniform_grid(
        input_mesh_path="./input_models/lowq.glb",
        output_dir="grid_parts_output",
        units=3  # e.g., 3x3x3 = 27 parts
    )


