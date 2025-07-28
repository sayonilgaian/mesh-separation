#!/usr/bin/env python3
"""
Fixed 3D Mesh Segmentation Pipeline
Addresses distance threshold and assignment issues.
"""

import numpy as np
import open3d as o3d
import trimesh
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import copy
from dataclasses import dataclass
from scipy.spatial.distance import cdist


@dataclass
class SegmentationMetrics:
    """Container for segmentation quality metrics."""
    alignment_quality: float
    parts_found: int
    vertices_assigned: int
    total_vertices: int
    coverage_ratio: float
    confidence_level: str


class ImprovedMeshSegmentationPipeline:
    def __init__(self, high_res_path: str, low_res_path: str, output_dir: str = "improved_segmented_parts"):
        """
        Initialize the improved segmentation pipeline.
        
        Args:
            high_res_path: Path to high-quality reference model (with separate parts)
            low_res_path: Path to low-quality single mesh model
            output_dir: Directory to save segmented parts
        """
        self.high_res_path = high_res_path
        self.low_res_path = low_res_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Loaded meshes
        self.high_res_mesh = None
        self.low_res_mesh = None
        self.reference_parts = {}
        self.segmented_parts = {}
        
        # Metrics
        self.metrics = None
        
    def step1_load_and_normalize_meshes(self, scale_normalize: bool = True, 
                                      align_centers: bool = True) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh]:
        """
        Step 1: Load and normalize meshes into the same coordinate space.
        
        Args:
            scale_normalize: Whether to normalize scale
            align_centers: Whether to align centers
            
        Returns:
            Tuple of (high_res_mesh, low_res_mesh)
        """
        print("Step 1: Loading and normalizing meshes...")
        
        # Load meshes
        self.high_res_mesh = self._load_mesh(self.high_res_path, "high-res")
        self.low_res_mesh = self._load_mesh(self.low_res_path, "low-res")
        
        # Validate meshes
        if len(self.high_res_mesh.vertices) == 0:
            raise ValueError("High-res mesh has no vertices")
        if len(self.low_res_mesh.vertices) == 0:
            raise ValueError("Low-res mesh has no vertices")
            
        print(f"High-res mesh: {len(self.high_res_mesh.vertices)} vertices, {len(self.high_res_mesh.triangles)} faces")
        print(f"Low-res mesh: {len(self.low_res_mesh.vertices)} vertices, {len(self.low_res_mesh.triangles)} faces")
        
        # Align centers
        if align_centers:
            high_center = self.high_res_mesh.get_center()
            low_center = self.low_res_mesh.get_center()
            translation = high_center - low_center
            self.low_res_mesh.translate(translation)
            print(f"Translated low-res mesh by: {translation}")
        
        # Scale normalization
        if scale_normalize:
            high_bbox = self.high_res_mesh.get_axis_aligned_bounding_box()
            low_bbox = self.low_res_mesh.get_axis_aligned_bounding_box()
            
            high_size = np.linalg.norm(high_bbox.get_extent())
            low_size = np.linalg.norm(low_bbox.get_extent())
            
            if high_size > 0 and low_size > 0:
                scale_factor = high_size / low_size
                self.low_res_mesh.scale(scale_factor, center=self.low_res_mesh.get_center())
                print(f"Scaled low-res mesh by factor: {scale_factor:.3f}")
        
        # Save normalized meshes
        o3d.io.write_triangle_mesh(str(self.output_dir / "high_res_normalized.ply"), self.high_res_mesh)
        o3d.io.write_triangle_mesh(str(self.output_dir / "low_res_normalized.ply"), self.low_res_mesh)
        
        print("✅ Step 1 completed: Meshes loaded and normalized")
        return self.high_res_mesh, self.low_res_mesh
    
    def step2_extract_reference_parts(self, min_triangles_ratio: float = 0.01) -> Dict[str, o3d.geometry.TriangleMesh]:
        """
        Step 2: Extract individual parts from the high-quality reference mesh.
        
        Args:
            min_triangles_ratio: Minimum ratio of triangles for a part to be kept
            
        Returns:
            Dictionary of part names to meshes
        """
        print("Step 2: Extracting reference parts...")
        
        if self.high_res_mesh is None:
            raise ValueError("High-res mesh not loaded. Run step1 first.")
        
        # Find connected components
        mesh_copy = copy.deepcopy(self.high_res_mesh)
        triangle_clusters, cluster_n_triangles, cluster_areas = mesh_copy.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        # Filter small clusters
        min_triangles = max(10, int(len(mesh_copy.triangles) * min_triangles_ratio))
        large_clusters = np.where(cluster_n_triangles > min_triangles)[0]
        
        print(f"Found {len(large_clusters)} significant parts in reference mesh")
        
        # Extract each part as a separate mesh
        self.reference_parts = {}
        
        for i, cluster_id in enumerate(large_clusters):
            # Get triangles for this cluster
            cluster_mask = triangle_clusters == cluster_id
            triangles_to_keep = np.where(cluster_mask)[0]
            
            if len(triangles_to_keep) == 0:
                continue
            
            # Create submesh
            part_mesh = mesh_copy.select_by_index(triangles_to_keep)
            
            if len(part_mesh.vertices) == 0:
                continue
            
            # Clean up part mesh
            part_mesh.remove_duplicated_vertices()
            part_mesh.remove_duplicated_triangles()
            part_mesh.remove_degenerate_triangles()
            part_mesh.remove_unreferenced_vertices()
            
            part_name = f"part_{i:02d}"
            self.reference_parts[part_name] = part_mesh
            
            # Save reference part
            o3d.io.write_triangle_mesh(str(self.output_dir / f"reference_{part_name}.ply"), part_mesh)
            
            print(f"  {part_name}: {len(part_mesh.vertices)} vertices, {len(part_mesh.triangles)} triangles")
        
        print(f"✅ Step 2 completed: Extracted {len(self.reference_parts)} reference parts")
        return self.reference_parts
    
    def _calculate_adaptive_threshold(self, low_vertices: np.ndarray, ref_vertices: np.ndarray) -> float:
        """Calculate adaptive distance threshold based on mesh characteristics."""
        # Get bounding box diagonal of low-res mesh
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(low_vertices)
        )
        bbox_diagonal = np.linalg.norm(bbox.get_extent())
        
        # Use a percentage of the bounding box diagonal
        adaptive_threshold = bbox_diagonal * 0.1  # 10% of diagonal
        
        # Also consider average edge length of low-res mesh
        if len(self.low_res_mesh.triangles) > 0:
            triangles = np.asarray(self.low_res_mesh.triangles)
            edge_lengths = []
            for tri in triangles[:min(1000, len(triangles))]:  # Sample for efficiency
                for i in range(3):
                    v1, v2 = tri[i], tri[(i+1) % 3]
                    edge_length = np.linalg.norm(low_vertices[v1] - low_vertices[v2])
                    edge_lengths.append(edge_length)
            
            if edge_lengths:
                avg_edge_length = np.mean(edge_lengths)
                edge_based_threshold = avg_edge_length * 3  # 3x average edge length
                adaptive_threshold = min(adaptive_threshold, edge_based_threshold)
        
        return max(adaptive_threshold, 0.001)  # Minimum threshold
    
    def step3_segment_using_actual_boundaries(self, search_radius: float = 0.05,
                                            min_distance_threshold: Optional[float] = None,
                                            use_watertight_test: bool = True,
                                            use_adaptive_threshold: bool = True) -> Dict[str, o3d.geometry.TriangleMesh]:
        """
        Step 3: Segment low-res mesh using actual boundaries of reference parts.
        
        Args:
            search_radius: Radius for finding nearby vertices
            min_distance_threshold: Minimum distance to consider a vertex as belonging to a part
            use_watertight_test: Use watertight volume test (slower but more accurate)
            use_adaptive_threshold: Use adaptive distance threshold calculation
            
        Returns:
            Dictionary of segmented parts
        """
        print("Step 3: Segmenting using actual part boundaries...")
        
        if not self.reference_parts or self.low_res_mesh is None:
            raise ValueError("Prerequisites not met. Run previous steps first.")
        
        low_vertices = np.asarray(self.low_res_mesh.vertices)
        vertex_assignments = {}  # vertex_index -> part_name
        vertex_distances = {}   # vertex_index -> distance_to_part
        
        print(f"Processing {len(low_vertices)} vertices against {len(self.reference_parts)} reference parts...")
        
        # Calculate adaptive threshold if needed
        if use_adaptive_threshold or min_distance_threshold is None:
            all_ref_vertices = np.vstack([np.asarray(part.vertices) for part in self.reference_parts.values()])
            adaptive_threshold = self._calculate_adaptive_threshold(low_vertices, all_ref_vertices)
            actual_threshold = min_distance_threshold if min_distance_threshold is not None else adaptive_threshold
            actual_threshold = max(actual_threshold, adaptive_threshold * 0.5)  # Use at least half of adaptive
            print(f"Using distance threshold: {actual_threshold:.4f} (adaptive: {adaptive_threshold:.4f})")
        else:
            actual_threshold = min_distance_threshold
            print(f"Using fixed distance threshold: {actual_threshold:.4f}")
        
        # For each reference part, find which low-res vertices belong to it
        for part_name, ref_part in self.reference_parts.items():
            print(f"  Processing {part_name}...")
            
            if use_watertight_test and self._is_mesh_watertight(ref_part):
                # Method 1: Watertight volume test (most accurate)
                inside_vertices = self._find_vertices_inside_mesh(low_vertices, ref_part)
                print(f"    Found {len(inside_vertices)} vertices inside watertight volume")
                
                for vertex_idx in inside_vertices:
                    vertex_assignments[vertex_idx] = part_name
                    vertex_distances[vertex_idx] = 0.0  # Inside volume
                    
            else:
                # Method 2: Distance-based assignment (fallback)
                print(f"    Using distance-based assignment (non-watertight mesh)")
                ref_vertices = np.asarray(ref_part.vertices)
                
                # Find closest reference vertices for each low-res vertex
                distances = cdist(low_vertices, ref_vertices)
                min_distances = distances.min(axis=1)
                
                # Assign vertices that are close enough
                close_vertices = np.where(min_distances <= actual_threshold)[0]
                print(f"    Found {len(close_vertices)} vertices within distance threshold")
                
                for vertex_idx in close_vertices:
                    current_distance = vertex_distances.get(vertex_idx, float('inf'))
                    if min_distances[vertex_idx] < current_distance:
                        vertex_assignments[vertex_idx] = part_name
                        vertex_distances[vertex_idx] = min_distances[vertex_idx]
                
                # Debug: Show distance statistics
                if len(min_distances) > 0:
                    print(f"    Distance stats - Min: {min_distances.min():.4f}, Max: {min_distances.max():.4f}, Mean: {min_distances.mean():.4f}")
        
        print(f"Total vertices assigned: {len(vertex_assignments)} / {len(low_vertices)}")
        
        # If very few vertices assigned, try with larger threshold
        if len(vertex_assignments) < len(low_vertices) * 0.1:  # Less than 10% assigned
            print("⚠️  Low assignment rate, trying with 2x threshold...")
            larger_threshold = actual_threshold * 2
            
            for part_name, ref_part in self.reference_parts.items():
                if self._is_mesh_watertight(ref_part):
                    continue  # Skip watertight parts, they were already processed
                    
                ref_vertices = np.asarray(ref_part.vertices)
                distances = cdist(low_vertices, ref_vertices)
                min_distances = distances.min(axis=1)
                
                close_vertices = np.where(min_distances <= larger_threshold)[0]
                new_assignments = 0
                
                for vertex_idx in close_vertices:
                    if vertex_idx not in vertex_assignments:  # Only assign unassigned vertices
                        vertex_assignments[vertex_idx] = part_name
                        vertex_distances[vertex_idx] = min_distances[vertex_idx]
                        new_assignments += 1
                
                if new_assignments > 0:
                    print(f"    {part_name}: +{new_assignments} vertices with larger threshold")
            
            print(f"After larger threshold: {len(vertex_assignments)} / {len(low_vertices)} vertices assigned")
        
        # Create segmented meshes
        self.segmented_parts = {}
        low_triangles = np.asarray(self.low_res_mesh.triangles)
        
        # Group vertices by assigned part
        part_vertex_sets = {}
        for vertex_idx, part_name in vertex_assignments.items():
            if part_name not in part_vertex_sets:
                part_vertex_sets[part_name] = set()
            part_vertex_sets[part_name].add(vertex_idx)
        
        # Create mesh for each part
        for part_name, vertex_set in part_vertex_sets.items():
            if len(vertex_set) == 0:
                continue
                
            print(f"  Creating mesh for {part_name} with {len(vertex_set)} vertices...")
            
            # Find triangles where all vertices belong to this part
            part_triangles = []
            for tri_idx, triangle in enumerate(low_triangles):
                if all(v_idx in vertex_set for v_idx in triangle):
                    part_triangles.append(tri_idx)
            
            # Also include triangles where at least 2 vertices belong to this part
            mixed_triangles = []
            for tri_idx, triangle in enumerate(low_triangles):
                vertices_in_part = sum(1 for v_idx in triangle if v_idx in vertex_set)
                if vertices_in_part >= 2 and tri_idx not in part_triangles:  # At least 2 out of 3 vertices
                    mixed_triangles.append(tri_idx)
            
            all_triangles = part_triangles + mixed_triangles
            print(f"    Triangles: {len(part_triangles)} complete + {len(mixed_triangles)} mixed = {len(all_triangles)} total")
            
            if all_triangles:
                # Create mesh with selected triangles
                part_mesh = self.low_res_mesh.select_by_index(all_triangles)
                
                if len(part_mesh.vertices) > 0:
                    # Clean up the mesh
                    part_mesh.remove_duplicated_vertices()
                    part_mesh.remove_duplicated_triangles()
                    part_mesh.remove_degenerate_triangles()
                    part_mesh.remove_unreferenced_vertices()
                    
                    if len(part_mesh.vertices) > 0:  # Still has vertices after cleanup
                        self.segmented_parts[part_name] = part_mesh
                        print(f"    ✅ Created {part_name}: {len(part_mesh.vertices)} vertices, {len(part_mesh.triangles)} triangles")
                    else:
                        print(f"    ❌ {part_name}: No vertices after cleanup")
                else:
                    print(f"    ❌ {part_name}: No vertices in selected triangles")
            else:
                print(f"    ❌ {part_name}: No triangles found")
        
        print(f"✅ Step 3 completed: Created {len(self.segmented_parts)} segmented parts")
        return self.segmented_parts
    
    def step4_refine_and_save_segments(self, remove_small_components: bool = True,
                                     min_component_ratio: float = 0.1,
                                     file_format: str = "ply") -> List[str]:
        """
        Step 4: Refine segments and save to files.
        
        Args:
            remove_small_components: Remove small disconnected components
            min_component_ratio: Minimum component size ratio to keep
            file_format: Output file format
            
        Returns:
            List of saved file paths
        """
        print("Step 4: Refining and saving segments...")
        
        if not self.segmented_parts:
            print("⚠️  No segmented parts found. Check if Step 3 succeeded.")
            # Instead of raising error, try to create parts from unassigned vertices
            return self._fallback_segmentation(file_format)
        
        refined_parts = {}
        saved_files = []
        
        for part_name, mesh in self.segmented_parts.items():
            print(f"  Refining {part_name}...")
            refined_mesh = copy.deepcopy(mesh)
            
            # Remove small components if requested
            if remove_small_components and len(refined_mesh.triangles) > 0:
                triangle_clusters, cluster_n_triangles, _ = refined_mesh.cluster_connected_triangles()
                triangle_clusters = np.asarray(triangle_clusters)
                cluster_n_triangles = np.asarray(cluster_n_triangles)
                
                if len(cluster_n_triangles) > 0:
                    # Keep largest component(s)
                    min_triangles = max(1, int(len(refined_mesh.triangles) * min_component_ratio))
                    large_clusters = np.where(cluster_n_triangles >= min_triangles)[0]
                    
                    if len(large_clusters) > 0:
                        triangles_to_keep = []
                        for cluster_id in large_clusters:
                            cluster_triangles = np.where(triangle_clusters == cluster_id)[0]
                            triangles_to_keep.extend(cluster_triangles.tolist())
                        
                        if triangles_to_keep:
                            refined_mesh = refined_mesh.select_by_index(triangles_to_keep)
                            refined_mesh.remove_unreferenced_vertices()
            
            # Final cleanup
            if len(refined_mesh.vertices) > 0:
                refined_mesh.remove_duplicated_vertices()
                refined_mesh.remove_duplicated_triangles()
                refined_mesh.remove_degenerate_triangles()
                
                if len(refined_mesh.vertices) > 0:
                    refined_parts[part_name] = refined_mesh
                    
                    # Save to file
                    filename = f"segmented_{part_name}.{file_format}"
                    filepath = self.output_dir / filename
                    
                    try:
                        success = o3d.io.write_triangle_mesh(str(filepath), refined_mesh)
                        if success:
                            saved_files.append(str(filepath))
                            print(f"    Saved: {filename} ({len(refined_mesh.vertices)} vertices)")
                        else:
                            print(f"    Failed to save: {filename}")
                    except Exception as e:
                        print(f"    Error saving {filename}: {e}")
        
        # Save metadata
        self._save_metadata(refined_parts, saved_files)
        
        print(f"✅ Step 4 completed: Saved {len(saved_files)} refined segments")
        return saved_files
    
    def _fallback_segmentation(self, file_format: str = "ply") -> List[str]:
        """Fallback method when main segmentation fails."""
        print("🔄 Running fallback segmentation...")
        
        if self.low_res_mesh is None:
            return []
        
        # Try to segment the low-res mesh directly into connected components
        mesh_copy = copy.deepcopy(self.low_res_mesh)
        triangle_clusters, cluster_n_triangles, _ = mesh_copy.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        saved_files = []
        min_triangles = max(1, len(mesh_copy.triangles) // 20)  # At least 5% of triangles
        large_clusters = np.where(cluster_n_triangles >= min_triangles)[0]
        
        print(f"Found {len(large_clusters)} components in low-res mesh for fallback")
        
        for i, cluster_id in enumerate(large_clusters):
            cluster_mask = triangle_clusters == cluster_id
            triangles_to_keep = np.where(cluster_mask)[0]
            
            if len(triangles_to_keep) > 0:
                part_mesh = mesh_copy.select_by_index(triangles_to_keep)
                part_mesh.remove_unreferenced_vertices()
                
                if len(part_mesh.vertices) > 0:
                    filename = f"fallback_part_{i:02d}.{file_format}"
                    filepath = self.output_dir / filename
                    
                    success = o3d.io.write_triangle_mesh(str(filepath), part_mesh)
                    if success:
                        saved_files.append(str(filepath))
                        print(f"  Saved fallback part: {filename}")
        
        return saved_files
    
    def _is_mesh_watertight(self, mesh: o3d.geometry.TriangleMesh) -> bool:
        """Check if a mesh is watertight (closed surface)."""
        try:
            return mesh.is_watertight()
        except:
            return False
    
    def _find_vertices_inside_mesh(self, vertices: np.ndarray, 
                                 mesh: o3d.geometry.TriangleMesh) -> List[int]:
        """
        Find which vertices are inside a watertight mesh.
        
        Args:
            vertices: Array of vertices to test
            mesh: Watertight mesh to test against
            
        Returns:
            List of vertex indices that are inside the mesh
        """
        try:
            # Convert to trimesh for ray casting
            trimesh_obj = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles)
            )
            
            # Use trimesh's contains method
            inside_mask = trimesh_obj.contains(vertices)
            inside_indices = np.where(inside_mask)[0].tolist()
            
            return inside_indices
            
        except Exception as e:
            print(f"    Warning: Volume test failed ({e}), falling back to distance method")
            return []
    
    def _compute_metrics(self) -> SegmentationMetrics:
        """Compute segmentation quality metrics."""
        if not self.segmented_parts:
            return SegmentationMetrics(0.0, 0, 0, 0, 0.0, "No Data")
        
        total_low_vertices = len(self.low_res_mesh.vertices)
        total_assigned_vertices = sum(len(mesh.vertices) for mesh in self.segmented_parts.values())
        coverage_ratio = total_assigned_vertices / total_low_vertices if total_low_vertices > 0 else 0.0
        
        # Simple alignment quality based on coverage
        if coverage_ratio >= 0.8:
            confidence = "High"
            alignment_quality = 0.9
        elif coverage_ratio >= 0.6:
            confidence = "Medium"  
            alignment_quality = 0.7
        elif coverage_ratio >= 0.4:
            confidence = "Low"
            alignment_quality = 0.5
        else:
            confidence = "Very Low"
            alignment_quality = 0.3
        
        return SegmentationMetrics(
            alignment_quality=alignment_quality,
            parts_found=len(self.reference_parts),
            vertices_assigned=total_assigned_vertices,
            total_vertices=total_low_vertices,
            coverage_ratio=coverage_ratio,
            confidence_level=confidence
        )
    
    def _save_metadata(self, refined_parts: Dict, saved_files: List[str]):
        """Save metadata about the segmentation process."""
        self.metrics = self._compute_metrics()
        
        metadata = {
            "original_files": {
                "high_res": self.high_res_path,
                "low_res": self.low_res_path
            },
            "reference_parts": {
                part_name: {
                    "vertices": len(mesh.vertices),
                    "triangles": len(mesh.triangles)
                }
                for part_name, mesh in self.reference_parts.items()
            },
            "segmented_parts": {
                part_name: {
                    "vertices": len(mesh.vertices),
                    "triangles": len(mesh.triangles),
                    "filename": f"segmented_{part_name}.ply"
                }
                for part_name, mesh in refined_parts.items()
            },
            "metrics": {
                "alignment_quality": self.metrics.alignment_quality,
                "parts_found": self.metrics.parts_found,
                "vertices_assigned": self.metrics.vertices_assigned,
                "total_vertices": self.metrics.total_vertices,
                "coverage_ratio": self.metrics.coverage_ratio,
                "confidence_level": self.metrics.confidence_level
            },
            "saved_files": saved_files
        }
        
        metadata_path = self.output_dir / "improved_segmentation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_mesh(self, file_path: str, mesh_type: str) -> o3d.geometry.TriangleMesh:
        """Load mesh from file with format detection."""
        try:
            mesh = o3d.io.read_triangle_mesh(file_path)
            if len(mesh.vertices) == 0:
                raise ValueError(f"No vertices loaded from {file_path}")
            return mesh
        except Exception as e:
            # Try with trimesh as fallback
            try:
                scene = trimesh.load(file_path)
                if hasattr(scene, 'geometry') and scene.geometry:
                    # Multiple geometries - combine them
                    geometries = [geom for geom in scene.geometry.values() 
                                if hasattr(geom, 'vertices') and len(geom.vertices) > 0]
                    if geometries:
                        combined = trimesh.util.concatenate(geometries) if len(geometries) > 1 else geometries[0]
                        return o3d.geometry.TriangleMesh(
                            o3d.utility.Vector3dVector(combined.vertices),
                            o3d.utility.Vector3iVector(combined.faces)
                        )
                elif hasattr(scene, 'vertices') and len(scene.vertices) > 0:
                    return o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(scene.vertices),
                        o3d.utility.Vector3iVector(scene.faces)
                    )
            except Exception as e2:
                pass
            
            raise ValueError(f"Failed to load {mesh_type} mesh from {file_path}: {e}")
    
    def run_improved_pipeline(self, scale_normalize: bool = True,
                            align_centers: bool = True,
                            search_radius: float = 0.05,
                            min_distance_threshold: Optional[float] = None,
                            use_watertight_test: bool = True,
                            use_adaptive_threshold: bool = True,
                            file_format: str = "ply") -> Tuple[List[str], SegmentationMetrics]:
        """
        Run the complete improved segmentation pipeline.
        
        Args:
            scale_normalize: Whether to normalize scale
            align_centers: Whether to align centers
            search_radius: Radius for vertex search
            min_distance_threshold: Distance threshold for assignment (None for adaptive)
            use_watertight_test: Use watertight volume testing
            use_adaptive_threshold: Use adaptive threshold calculation
            file_format: Output file format
            
        Returns:
            Tuple of (saved_files, metrics)
        """
        print("🚀 Starting Improved Mesh Segmentation Pipeline...")
        print("=" * 60)
        
        # Step 1: Load and normalize
        self.step1_load_and_normalize_meshes(scale_normalize, align_centers)
        
        # Step 2: Extract reference parts  
        self.step2_extract_reference_parts()
        
        # Step 3: Segment using actual boundaries
        self.step3_segment_using_actual_boundaries(
            search_radius=search_radius,
            min_distance_threshold=min_distance_threshold,
            use_watertight_test=use_watertight_test,
            use_adaptive_threshold=use_adaptive_threshold
        )
        
        # Step 4: Refine and save
        saved_files = self.step4_refine_and_save_segments(file_format=file_format)
        
        # Compute final metrics
        metrics = self._compute_metrics()
        self.metrics = metrics
        
        print("\n" + "=" * 60)
        print("🎉 Improved Pipeline Completed!")
        print(f"📊 Results Summary:")
        print(f"   • Reference parts found: {metrics.parts_found}")
        print(f"   • Segments created: {len(self.segmented_parts)}")
        print(f"   • Files saved: {len(saved_files)}")
        print(f"   • Vertex coverage: {metrics.coverage_ratio:.1%}")
        print(f"   • Confidence level: {metrics.confidence_level}")
        print(f"📁 Output directory: {self.output_dir}")
        
        return saved_files, metrics
    
    def get_segmentation_report(self) -> str:
        """Generate a comprehensive segmentation report."""
        if not self.metrics:
            return "No segmentation metrics available. Run the pipeline first."
        
        report = f"""
IMPROVED MESH SEGMENTATION REPORT
=================================

INPUT FILES:
- High-resolution reference: {self.high_res_path}
- Low-resolution target: {self.low_res_path}

SEGMENTATION RESULTS:
- Reference parts found: {self.metrics.parts_found}
- Target segments created: {len(self.segmented_parts)}
- Vertices assigned: {self.metrics.vertices_assigned:,} / {self.metrics.total_vertices:,}
- Coverage ratio: {self.metrics.coverage_ratio:.1%}

QUALITY ASSESSMENT:
- Alignment quality: {self.metrics.alignment_quality:.3f}
- Confidence level: {self.metrics.confidence_level}

RECOMMENDATIONS:
"""
        
        if self.metrics.coverage_ratio >= 0.8:
            report += "✅ Excellent coverage - segmentation results look good!\n"
        elif self.metrics.coverage_ratio >= 0.6:
            report += "✅ Good coverage - results should be usable.\n"
        elif self.metrics.coverage_ratio >= 0.4:
            report += "⚠️  Moderate coverage - check individual parts.\n"
        else:
            report += "⚠️  Low coverage - meshes may be too different.\n"
        
        report += f"\nOutput saved to: {self.output_dir}\n"
        return report


# Example usage with debugging
def main():
    """Example usage of the improved mesh segmentation pipeline."""
    
    # Example paths
    high_res_path = "./input_models/highq.ply"
    low_res_path = "./input_models/lowq.ply"
    
    # Create pipeline
    pipeline = ImprovedMeshSegmentationPipeline(
        high_res_path=high_res_path,
        low_res_path=low_res_path,
        output_dir="improved_segmented_parts"
    )
    
    # Run pipeline with adaptive settings
    try:
        saved_files, metrics = pipeline.run_improved_pipeline(
            scale_normalize=True,
            align_centers=True,
            search_radius=0.05,
            min_distance_threshold=None,  # Use adaptive threshold
            use_watertight_test=True,
            use_adaptive_threshold=True,
            file_format="ply"
        )
        
        # Print report
        print("\n" + pipeline.get_segmentation_report())
        
        print(f"\n🎯 Segmented mesh into {len(saved_files)} parts:")
        for file_path in saved_files:
            print(f"  - {file_path}")
            
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()