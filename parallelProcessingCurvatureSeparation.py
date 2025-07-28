import numpy as np
import open3d as o3d
import copy
from pathlib import Path
from collections import deque
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import trimesh
from trimesh import Trimesh, Scene

def find_parts(mesh, angle_threshold_deg):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    tri_normals = np.asarray(mesh.triangle_normals)
    n_tri = len(triangles)

    # Build adjacency
    tri_adj = [[] for _ in range(n_tri)]
    edge_map = {}
    for tidx, tri in enumerate(triangles):
        for i in range(3):
            key = tuple(sorted((int(tri[i]), int(tri[(i+1)%3]))))
            edge_map.setdefault(key, []).append(tidx)
    for neis in edge_map.values():
        if len(neis) == 2:
            a,b = neis
            tri_adj[a].append(b)
            tri_adj[b].append(a)

    thresh = np.cos(np.deg2rad(angle_threshold_deg))
    sharp = {}
    for i in range(n_tri):
        for j in tri_adj[i]:
            sharp[(i,j)] = np.dot(tri_normals[i], tri_normals[j]) < thresh

    visited = np.zeros(n_tri, dtype=bool)
    parts = []
    for start in range(n_tri):
        if visited[start]:
            continue
        q = deque([start])
        visited[start] = True
        comp = [start]
        while q:
            cur = q.popleft()
            for nei in tri_adj[cur]:
                if visited[nei]: continue
                if sharp.get((cur,nei), False) or sharp.get((nei,cur), False):
                    continue
                visited[nei] = True
                comp.append(nei)
                q.append(nei)
        parts.append(comp)
    return parts

def process_part(worker_args):
    input_path, part_idx, tri_indices = worker_args
    mesh = o3d.io.read_triangle_mesh(input_path)
    mesh.compute_vertex_normals()
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    n_tri = len(np.asarray(mesh.triangles))

    mesh_copy = copy.deepcopy(mesh)
    to_remove = list(set(range(n_tri)) - set(tri_indices))
    mesh_copy.remove_triangles_by_index(to_remove)
    mesh_copy.remove_unreferenced_vertices()
    mesh_copy.remove_duplicated_vertices()
    mesh_copy.remove_degenerate_triangles()

    if not mesh_copy.has_triangles():
        return None

    tm = Trimesh(vertices=np.asarray(mesh_copy.vertices),
                 faces=np.asarray(mesh_copy.triangles),
                 vertex_normals=np.asarray(mesh_copy.vertex_normals),
                 process=False)
    return (part_idx, tm)

def segment_and_export_parallel(input_path, output_glb,
                                angle_threshold_deg=30.0, workers=4):
    mesh = o3d.io.read_triangle_mesh(input_path)
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh: {input_path}")
    mesh.compute_vertex_normals()
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    parts = find_parts(mesh, angle_threshold_deg)
    print(f"Found {len(parts)} parts with threshold {angle_threshold_deg}°")

    scene = Scene()

    args_list = [(input_path, idx, parts[idx]) for idx in range(len(parts))]

    with Pool(workers) as pool:
        for res in tqdm(pool.imap(process_part, args_list), total=len(parts), desc="Processing parts"):
            if res is None:
                continue
            idx, tm = res
            scene.add_geometry(tm, node_name=f"part_{idx:04d}")

    Path(output_glb).parent.mkdir(parents=True, exist_ok=True)
    scene.export(output_glb)
    print(f"\n🎉 Done: exported scene with {len(scene.geometry)} parts → {output_glb}")

if __name__ == "__main__":
    segment_and_export_parallel(
        input_path="./input_models/lowq.glb",
        output_glb="curvature_segments_scene.glb",
        angle_threshold_deg=150.0,
        workers=4
    )
