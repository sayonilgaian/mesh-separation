import numpy as np
import open3d as o3d
import copy
from pathlib import Path
from collections import deque

def segment_by_curvature_joints(input_path, output_dir, angle_threshold_deg=30.0):
    """
    Segments mesh along sharp-curvature joints (angle between adjacent triangle normals).
    Saves each part as a .glb in the same global coordinate frame.
    """
    mesh = o3d.io.read_triangle_mesh(input_path)
    if not mesh.has_triangles():
        raise ValueError("Mesh is empty")
    mesh.compute_vertex_normals()
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    # Compute triangle normals
    tri_normals = np.asarray(mesh.triangle_normals)
    n_tri = len(triangles)

    # Build adjacency: triangle index -> neighboring triangle indices
    tri_adj = [[] for _ in range(n_tri)]
    # edge map: normalized edge (min,max) → triangles
    edge_map = {}
    for tidx, tri in enumerate(triangles):
        edges = [(tri[i], tri[(i+1)%3]) for i in range(3)]
        for v0,v1 in edges:
            key = tuple(sorted((int(v0), int(v1))))
            edge_map.setdefault(key, []).append(tidx)
    for neis in edge_map.values():
        if len(neis) == 2:
            a,b = neis
            tri_adj[a].append(b)
            tri_adj[b].append(a)

    # Compute adjacency mask: whether edge is sharp
    sharp_mask = {}
    thresh = np.cos(np.deg2rad(angle_threshold_deg))
    for i in range(n_tri):
        for j in tri_adj[i]:
            dot = np.dot(tri_normals[i], tri_normals[j])
            if dot < thresh:
                sharp_mask[(i,j)] = True
            else:
                sharp_mask[(i,j)] = False

    # Connected component grouping on triangles
    visited = np.zeros(n_tri, dtype=bool)
    part_labels = np.full(n_tri, -1, dtype=int)
    parts = {}
    current_label = 0

    for i in range(n_tri):
        if visited[i]:
            continue
        # BFS
        q = deque([i])
        visited[i] = True
        parts[current_label] = [i]
        part_labels[i] = current_label
        while q:
            cur = q.popleft()
            for nei in tri_adj[cur]:
                if visited[nei]:
                    continue
                if sharp_mask.get((cur,nei), False) or sharp_mask.get((nei,cur), False):
                    continue
                visited[nei] = True
                part_labels[nei] = current_label
                parts[current_label].append(nei)
                q.append(nei)
        current_label += 1

    print(f"Found {len(parts)} parts with angle threshold {angle_threshold_deg}°")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved = 0
    total = len(parts)

    for label, tri_indices in parts.items():
        if len(tri_indices) == 0:
            continue
        mesh_copy = copy.deepcopy(mesh)
        remove = list(set(range(n_tri)) - set(tri_indices))
        mesh_copy.remove_triangles_by_index(remove)
        mesh_copy.remove_unreferenced_vertices()
        mesh_copy.remove_duplicated_vertices()
        mesh_copy.remove_degenerate_triangles()
        if not mesh_copy.has_triangles():
            continue
        fname = f"part_{label:03d}.glb"
        out = Path(output_dir) / fname
        ok = o3d.io.write_triangle_mesh(str(out), mesh_copy, write_ascii=False, write_triangle_uvs=True)
        if ok:
            print(f"✅ Saved {fname} ({len(mesh_copy.triangles)} tris)")
            saved += 1
        else:
            print(f"❌ Failed {fname}")

    print(f"\n🎉 Done: saved {saved}/{total} parts to {output_dir}")

if __name__ == "__main__":
    segment_by_curvature_joints(
    input_path="./input_models/lowq.glb",
    output_dir="curvature_segments",
    angle_threshold_deg=150.0
)



