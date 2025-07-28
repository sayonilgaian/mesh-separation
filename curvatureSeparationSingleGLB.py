import numpy as np
import open3d as o3d
import copy
from pathlib import Path
from collections import deque

import trimesh
from trimesh import Trimesh, Scene

def segment_and_export_scene(input_path, output_glb, angle_threshold_deg=30.0):
    mesh = o3d.io.read_triangle_mesh(input_path)
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh or mesh is empty: {input_path}")
    mesh.compute_vertex_normals()
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    tri_normals = np.asarray(mesh.triangle_normals)
    n_tri = len(triangles)

    # Build triangle adjacency
    tri_adj = [[] for _ in range(n_tri)]
    edge_map = {}
    for tidx, tri in enumerate(triangles):
        for i in range(3):
            v0, v1 = tri[i], tri[(i+1)%3]
            key = tuple(sorted((int(v0), int(v1))))
            edge_map.setdefault(key, []).append(tidx)
    for neis in edge_map.values():
        if len(neis)==2:
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
        if visited[start]: continue
        q = deque([start]); visited[start]=True
        comp = [start]
        while q:
            cur = q.popleft()
            for nei in tri_adj[cur]:
                if visited[nei]: continue
                if sharp.get((cur,nei), False) or sharp.get((nei,cur), False):
                    continue
                visited[nei]=True
                comp.append(nei)
                q.append(nei)
        parts.append(comp)

    print(f"Found {len(parts)} segmented parts (threshold={angle_threshold_deg}°)")

    scene = Scene()
    saved = 0
    for label, tri_indices in enumerate(parts):
        print(f"Processing part {label+1}/{len(parts)} with {len(tri_indices)} triangles...")
        mesh_copy = copy.deepcopy(mesh)
        remove = list(set(range(n_tri)) - set(tri_indices))
        mesh_copy.remove_triangles_by_index(remove)
        mesh_copy.remove_unreferenced_vertices()
        mesh_copy.remove_duplicated_vertices()
        mesh_copy.remove_degenerate_triangles()
        if not mesh_copy.has_triangles(): 
            print(" → part is empty after cleanup, skipping")
            continue

        # convert to trimesh
        tm = Trimesh(vertices=np.asarray(mesh_copy.vertices),
                     faces=np.asarray(mesh_copy.triangles),
                     vertex_normals=np.asarray(mesh_copy.vertex_normals),
                     process=False)
        scene.add_geometry(tm, node_name=f"part_{label:03d}")
        saved += 1

    Path(output_glb).parent.mkdir(parents=True, exist_ok=True)
    scene.export(output_glb)
    print(f"\n🎉 Done: exported {saved}/{len(parts)} parts into single scene: {output_glb}")

if __name__ == "__main__":
    segment_and_export_scene(
        input_path="./input_models/lowq.glb",
        output_glb="curvature_segments_scene.glb",
        angle_threshold_deg=210.0
    )
