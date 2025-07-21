#!/usr/bin/env python3
"""
3D Mesh Segmentation Pipeline
Split a low-quality single-mesh 3D object into meaningful parts using a high-quality reference model.
"""

import numpy as np
import open3d as o3d
import trimesh
import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import copy


class MeshSegmentationPipeline:
    def __init__(self, high_res_path: str, low_res_path: str, output_dir: str = "segmented_parts"):
        """
        Initialize the segmentation pipeline.
        
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
        self.part_bounding_boxes = {}
        self.segmented_parts = {}
        
    def step1_load_and_normalize_meshes(self, scale_normalize: bool = True, 
                                      align_centers: bool = True) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh]:
        """
        Step 1: Load and normalize the meshes into the same coordinate space.
        
        Args:
            scale_normalize: Whether to normalize scale
            align_centers: Whether to align centers
            
        Returns:
            Tuple of (high_res_mesh, low_res_mesh)
        """
        print("Step 1: Loading and normalizing meshes...")
        
        # Load high-res mesh
        self.high_res_mesh = self._load_mesh(self.high_res_path, "high-res")
        
        # Load low-res mesh
        self.low_res_mesh = self._load_mesh(self.low_res_path, "low-res")
        
        # Validate meshes loaded successfully
        if len(self.high_res_mesh.vertices) == 0:
            raise ValueError("High-res mesh has no vertices")
        if len(self.low_res_mesh.vertices) == 0:
            raise ValueError("Low-res mesh has no vertices")
            
        print(f"High-res mesh: {len(self.high_res_mesh.vertices)} vertices, {len(self.high_res_mesh.triangles)} faces")
        print(f"Low-res mesh: {len(self.low_res_mesh.vertices)} vertices, {len(self.low_res_mesh.triangles)} faces")
        
        # Normalize scale if requested
        if scale_normalize:
            high_bbox = self.high_res_mesh.get_axis_aligned_bounding_box()
            low_bbox = self.low_res_mesh.get_axis_aligned_bounding_box()
            
            high_size = np.linalg.norm(high_bbox.get_extent())
            low_size = np.linalg.norm(low_bbox.get_extent())
            
            if high_size > 0 and low_size > 0:
                scale_factor = high_size / low_size
                self.low_res_mesh.scale(scale_factor, center=self.low_res_mesh.get_center())
                print(f"Scaled low-res mesh by factor: {scale_factor:.3f}")
        
        # Align centers if requested
        if align_centers:
            high_center = self.high_res_mesh.get_center()
            low_center = self.low_res_mesh.get_center()
            translation = high_center - low_center
            self.low_res_mesh.translate(translation)
            print(f"Translated low-res mesh by: {translation}")
        
        # Save normalized meshes for debugging
        o3d.io.write_triangle_mesh(str(self.output_dir / "high_res_normalized.ply"), self.high_res_mesh)
        o3d.io.write_triangle_mesh(str(self.output_dir / "low_res_normalized.ply"), self.low_res_mesh)
        
        print("✅ Step 1 completed: Meshes loaded and normalized")
        return self.high_res_mesh, self.low_res_mesh
    
    def step2_identify_part_bounding_volumes(self, use_obb: bool = False, 
                                           expansion_factor: float = 0.1) -> Dict[str, o3d.geometry.AxisAlignedBoundingBox]:
        """
        Step 2: Extract bounding boxes for parts in the high-quality mesh.
        
        Args:
            use_obb: Use Oriented Bounding Box instead of Axis-Aligned
            expansion_factor: Factor to expand bounding boxes (0.1 = 10% expansion)
            
        Returns:
            Dictionary of part names to bounding boxes
        """
        print("Step 2: Identifying part bounding volumes...")
        
        if self.high_res_mesh is None:
            raise ValueError("High-res mesh not loaded. Run step1 first.")
        
        # Method 1: If high-res mesh is actually multiple separate components
        mesh_copy = copy.deepcopy(self.high_res_mesh)
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_copy.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        
        # Filter out small clusters (noise)
        min_triangles = max(10, len(mesh_copy.triangles) // 100)  # At least 1% of total triangles
        large_clusters = []
        for i in range(len(cluster_n_triangles)):
            if cluster_n_triangles[i] > min_triangles:
                large_clusters.append(i)
        
        print(f"Found {len(large_clusters)} significant parts in high-res mesh")
        
        # Extract bounding boxes for each part
        self.part_bounding_boxes = {}
        
        for i, cluster_id in enumerate(large_clusters):
            # Extract triangles for this cluster
            cluster_triangles = triangle_clusters == cluster_id
            
            # Create submesh for this part
            triangles_to_keep = []
            for j, keep in enumerate(cluster_triangles):
                if keep:
                    triangles_to_keep.append(j)
            
            if not triangles_to_keep:
                continue
                
            # Create new mesh with only this part's triangles
            part_mesh = mesh_copy.select_by_index(triangles_to_keep)
            
            if len(part_mesh.vertices) == 0:
                continue
            
            # Get bounding box
            if use_obb:
                bbox = part_mesh.get_oriented_bounding_box()
            else:
                bbox = part_mesh.get_axis_aligned_bounding_box()
            
            # Expand bounding box
            if expansion_factor > 0:
                center = bbox.get_center()
                extent = bbox.get_extent()
                expanded_extent = extent * (1 + expansion_factor)
                
                # Create new expanded AABB
                min_bound = center - expanded_extent / 2
                max_bound = center + expanded_extent / 2
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            
            part_name = f"part_{i:02d}"
            self.part_bounding_boxes[part_name] = bbox
            
            # Save part mesh for debugging
            o3d.io.write_triangle_mesh(str(self.output_dir / f"highres_{part_name}.ply"), part_mesh)
            print(f"  {part_name}: {len(part_mesh.vertices)} vertices")
        
        # Save bounding box visualization
        self._visualize_bounding_boxes()
        
        print(f"✅ Step 2 completed: Found {len(self.part_bounding_boxes)} parts")
        return self.part_bounding_boxes
    
    def step3_crop_low_quality_mesh(self) -> Dict[str, o3d.geometry.TriangleMesh]:
        """
        Step 3: Use bounding boxes to extract part-like sub-meshes from low-res model.
        
        Returns:
            Dictionary of part names to cropped meshes
        """
        print("Step 3: Cropping low-quality mesh based on bounding boxes...")
        
        if self.low_res_mesh is None or not self.part_bounding_boxes:
            raise ValueError("Prerequisites not met. Run steps 1 and 2 first.")
        
        cropped_meshes = {}
        
        for part_name, bbox in self.part_bounding_boxes.items():
            print(f"  Processing {part_name}...")
            
            # Crop the low-res mesh using the bounding box
            try:
                cropped_mesh = self.low_res_mesh.crop(bbox)
                
                if len(cropped_mesh.vertices) > 0:
                    cropped_meshes[part_name] = cropped_mesh
                    print(f"    Cropped: {len(cropped_mesh.vertices)} vertices, {len(cropped_mesh.triangles)} triangles")
                else:
                    print(f"    Warning: No geometry found in bounding box for {part_name}")
                    
            except Exception as e:
                print(f"    Error cropping {part_name}: {e}")
                continue
        
        self.segmented_parts = cropped_meshes
        
        print(f"✅ Step 3 completed: Successfully cropped {len(cropped_meshes)} parts")
        return cropped_meshes
    
    def step4_refine_mesh_segments(self, remove_small_components: bool = True,
                                 min_component_size: int = 10,
                                 repair_mesh: bool = True) -> Dict[str, o3d.geometry.TriangleMesh]:
        """
        Step 4: Clean up and refine the cropped mesh segments.
        
        Args:
            remove_small_components: Remove small disconnected components
            min_component_size: Minimum triangles for a component to keep
            repair_mesh: Attempt to repair mesh issues
            
        Returns:
            Dictionary of refined mesh segments
        """
        print("Step 4: Refining mesh segments...")
        
        if not self.segmented_parts:
            raise ValueError("No segmented parts found. Run step 3 first.")
        
        refined_parts = {}
        
        for part_name, mesh in self.segmented_parts.items():
            print(f"  Refining {part_name}...")
            refined_mesh = copy.deepcopy(mesh)
            
            # Remove small disconnected components
            if remove_small_components:
                triangle_clusters, cluster_n_triangles, _ = refined_mesh.cluster_connected_triangles()
                triangle_clusters = np.asarray(triangle_clusters)
                
                # Find largest cluster
                largest_cluster_id = np.argmax(cluster_n_triangles)
                if cluster_n_triangles[largest_cluster_id] >= min_component_size:
                    # Keep only triangles from largest cluster
                    triangles_to_keep = triangle_clusters == largest_cluster_id
                    triangles_to_remove = []
                    for i, keep in enumerate(triangles_to_keep):
                        if not keep:
                            triangles_to_remove.append(i)
                    
                    refined_mesh.remove_triangles_by_index(triangles_to_remove)
                    refined_mesh.remove_unreferenced_vertices()
                    
                    print(f"    Removed {len(triangles_to_remove)} triangles from small components")
            
            # Basic mesh repair
            if repair_mesh and len(refined_mesh.vertices) > 0:
                # Remove duplicated vertices
                refined_mesh.remove_duplicated_vertices()
                
                # Remove duplicated triangles  
                refined_mesh.remove_duplicated_triangles()
                
                # Remove degenerate triangles
                refined_mesh.remove_degenerate_triangles()
                
                # Remove non-manifold edges (optional, can be aggressive)
                # refined_mesh.remove_non_manifold_edges()
            
            if len(refined_mesh.vertices) > 0:
                refined_parts[part_name] = refined_mesh
                print(f"    Final: {len(refined_mesh.vertices)} vertices, {len(refined_mesh.triangles)} triangles")
            else:
                print(f"    Warning: {part_name} has no vertices after refinement")
        
        self.segmented_parts = refined_parts
        
        print(f"✅ Step 4 completed: Refined {len(refined_parts)} parts")
        return refined_parts
    
    def step5_save_separated_objects(self, file_format: str = "ply") -> List[str]:
        """
        Step 5: Save each mesh segment as separate files.
        
        Args:
            file_format: Output format ('ply', 'obj', 'stl')
            
        Returns:
            List of saved file paths
        """
        print("Step 5: Saving separated objects...")
        
        if not self.segmented_parts:
            raise ValueError("No segmented parts found. Run previous steps first.")
        
        saved_files = []
        
        for part_name, mesh in self.segmented_parts.items():
            filename = f"lowres_{part_name}.{file_format}"
            filepath = self.output_dir / filename
            
            try:
                success = o3d.io.write_triangle_mesh(str(filepath), mesh)
                if success:
                    saved_files.append(str(filepath))
                    print(f"  Saved: {filename}")
                else:
                    print(f"  Failed to save: {filename}")
            except Exception as e:
                print(f"  Error saving {filename}: {e}")
        
        # Save metadata
        metadata = {
            "original_files": {
                "high_res": self.high_res_path,
                "low_res": self.low_res_path
            },
            "parts": {
                part_name: {
                    "vertices": len(mesh.vertices),
                    "triangles": len(mesh.triangles),
                    "filename": f"lowres_{part_name}.{file_format}"
                }
                for part_name, mesh in self.segmented_parts.items()
            },
            "bounding_boxes": {
                part_name: {
                    "min_bound": bbox.min_bound.tolist(),
                    "max_bound": bbox.max_bound.tolist()
                }
                for part_name, bbox in self.part_bounding_boxes.items()
            }
        }
        
        metadata_path = self.output_dir / "segmentation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Step 5 completed: Saved {len(saved_files)} files")
        print(f"📁 Output directory: {self.output_dir}")
        return saved_files
    
    def _load_mesh(self, file_path: str, mesh_type: str) -> o3d.geometry.TriangleMesh:
        """
        Helper method to load meshes with support for various formats including FBX.
        
        Args:
            file_path: Path to mesh file
            mesh_type: Description for error messages
            
        Returns:
            Open3D triangle mesh
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.fbx':
            return self._load_fbx_mesh(file_path, mesh_type)
        else:
            return self._load_standard_mesh(file_path, mesh_type)
    
    def _load_fbx_mesh(self, file_path: str, mesh_type: str) -> o3d.geometry.TriangleMesh:
        """
        Load FBX files using multiple fallback methods.
        """
        print(f"  Loading FBX file: {file_path}")
        
        # Method 1: Try trimesh with FBX support
        try:
            scene = trimesh.load(file_path)
            return self._process_trimesh_scene(scene, file_path)
        except Exception as e:
            print(f"  ❌ Trimesh failed: {e}")
        
        # Method 2: Try with Blender conversion (if available)
        try:
            return self._load_fbx_via_blender(file_path, mesh_type)
        except Exception as e:
            print(f"  ❌ Blender conversion failed: {e}")
        
        # Method 3: Try with FBX SDK (if available)
        try:
            return self._load_fbx_via_sdk(file_path)
        except Exception as e:
            print(f"  ❌ FBX SDK failed: {e}")
        
        # Final fallback: Suggest conversion
        raise ValueError(
            f"❌ Failed to load FBX file: {file_path}\n"
            f"💡 Solutions:\n"
            f"   1. Install FBX support: pip install 'trimesh[all]'\n"
            f"   2. Convert to OBJ/PLY using Blender\n"
            f"   3. Use online converter (e.g., aspose.com)\n"
            f"   4. Export from original software as OBJ/PLY"
        )
    
    def _load_fbx_via_blender(self, fbx_path: str, mesh_type: str) -> o3d.geometry.TriangleMesh:
        """
        Convert FBX to OBJ using Blender command line, then load the OBJ.
        """
        # Check if Blender is available
        blender_cmd = self._find_blender_executable()
        if not blender_cmd:
            raise Exception("Blender not found in PATH")
        
        print(f"  Converting FBX to OBJ using Blender...")
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_obj:
            temp_obj_path = temp_obj.name
        
        # Blender script to convert FBX to OBJ
        blender_script = f'''
import bpy
import sys

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Import FBX
try:
    bpy.ops.import_scene.fbx(filepath="{fbx_path}")
    print("✅ FBX imported successfully")
except Exception as e:
    print(f"❌ FBX import failed: {{e}}")
    sys.exit(1)

# Export as OBJ
try:
    bpy.ops.export_scene.obj(
        filepath="{temp_obj_path}",
        use_selection=False,
        use_materials=False,
        use_uvs=False,
        use_normals=True
    )
    print("✅ OBJ exported successfully")
except Exception as e:
    print(f"❌ OBJ export failed: {{e}}")
    sys.exit(1)
'''
        
        # Write script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
            script_file.write(blender_script)
            script_path = script_file.name
        
        try:
            # Run Blender
            result = subprocess.run([
                blender_cmd, '--background', '--python', script_path
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Load the converted OBJ
                mesh = o3d.io.read_triangle_mesh(temp_obj_path)
                if len(mesh.vertices) > 0:
                    print(f"  ✅ Successfully converted FBX to OBJ: {len(mesh.vertices)} vertices")
                    return mesh
                else:
                    raise Exception("Converted OBJ file is empty")
            else:
                raise Exception(f"Blender failed: {result.stderr}")
                
        finally:
            # Cleanup temp files
            try:
                os.unlink(temp_obj_path)
                os.unlink(script_path)
            except:
                pass
    
    def _load_fbx_via_sdk(self, file_path: str) -> o3d.geometry.TriangleMesh:
        """
        Load FBX using the official Autodesk FBX SDK (if available).
        """
        try:
            import fbx
        except ImportError:
            raise Exception("FBX SDK not available. Install with: pip install fbx-python-sdk")
        
        print(f"  Loading FBX with official SDK...")
        # This is a placeholder - full FBX SDK implementation is complex
        # For now, just raise an exception to move to next fallback
        raise Exception("FBX SDK loader not fully implemented yet")
    
    def _find_blender_executable(self) -> Optional[str]:
        """Find Blender executable in common locations."""
        common_paths = [
            'blender',  # In PATH
            '/usr/bin/blender',  # Linux
            '/usr/local/bin/blender',  # Linux/macOS
            '/Applications/Blender.app/Contents/MacOS/Blender',  # macOS
            'C:\\Program Files\\Blender Foundation\\Blender 3.6\\blender.exe',  # Windows
            'C:\\Program Files\\Blender Foundation\\Blender 4.0\\blender.exe',  # Windows
        ]
        
        for path in common_paths:
            try:
                result = subprocess.run([path, '--version'], capture_output=True, timeout=5)
                if result.returncode == 0:
                    return path
            except:
                continue
        
        return None
    
    def _process_trimesh_scene(self, scene, file_path: str) -> o3d.geometry.TriangleMesh:
        """Process a trimesh scene object into an Open3D mesh."""
        if hasattr(scene, 'geometry') and scene.geometry:
            # Multiple geometries - combine them
            print(f"  Found {len(scene.geometry)} geometries in file")
            geometries = []
            
            for name, geom in scene.geometry.items():
                if hasattr(geom, 'vertices') and len(geom.vertices) > 0:
                    print(f"    Geometry '{name}': {len(geom.vertices)} vertices")
                    geometries.append(geom)
            
            if geometries:
                if len(geometries) == 1:
                    combined = geometries[0]
                else:
                    # Combine all geometries
                    combined = trimesh.util.concatenate(geometries)
                
                # Convert to Open3D
                mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(combined.vertices),
                    o3d.utility.Vector3iVector(combined.faces)
                )
                
                # Copy vertex colors if they exist
                if hasattr(combined, 'visual') and hasattr(combined.visual, 'vertex_colors'):
                    colors = combined.visual.vertex_colors[:, :3] / 255.0  # Normalize to [0,1]
                    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                
                return mesh
            else:
                raise ValueError(f"No valid geometries found in file: {file_path}")
                
        elif hasattr(scene, 'vertices') and len(scene.vertices) > 0:
            # Single geometry
            print(f"  Single geometry: {len(scene.vertices)} vertices")
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(scene.vertices),
                o3d.utility.Vector3iVector(scene.faces)
            )
            return mesh
        else:
            raise ValueError(f"No geometry data found in file: {file_path}")
    
    def _load_standard_mesh(self, file_path: str, mesh_type: str) -> o3d.geometry.TriangleMesh:
        """
        Load standard format meshes (PLY, OBJ, STL, etc.)
        """
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext in ['.glb', '.gltf']:
                print(f"  Loading GLTF/GLB file: {file_path}")
                scene = trimesh.load(file_path)
                return self._process_trimesh_scene(scene, file_path)
            else:
                # Use Open3D for standard formats (PLY, OBJ, STL, etc.)
                print(f"  Loading {file_ext.upper()} file with Open3D: {file_path}")
                mesh = o3d.io.read_triangle_mesh(file_path)
                
                if len(mesh.vertices) == 0:
                    raise ValueError(f"No vertices loaded from {file_path}")
                    
                return mesh
                
        except Exception as e:
            print(f"❌ Error loading {mesh_type} mesh from {file_path}: {e}")
            
            # Try alternative loading method
            print("  Trying alternative loading method...")
            try:
                scene = trimesh.load(file_path)
                return self._process_trimesh_scene(scene, file_path)
            except Exception as e2:
                print(f"  ❌ Fallback also failed: {e2}")
            
            raise ValueError(f"Failed to load {mesh_type} mesh from {file_path}. Ensure file exists and is a valid 3D model.")
    
    def _visualize_bounding_boxes(self):
        """Helper method to visualize bounding boxes for debugging."""
        vis_geometries = [self.high_res_mesh]
        
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
        
        for i, (part_name, bbox) in enumerate(self.part_bounding_boxes.items()):
            # Create wireframe box for visualization
            bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
            bbox_lines.paint_uniform_color(colors[i % len(colors)])
            vis_geometries.append(bbox_lines)
        
        # Save visualization scene
        print("  Saved bounding box visualization")
    
    def run_complete_pipeline(self, scale_normalize: bool = True, 
                            align_centers: bool = True,
                            expansion_factor: float = 0.1,
                            file_format: str = "ply") -> List[str]:
        """
        Run the complete segmentation pipeline.
        
        Args:
            scale_normalize: Whether to normalize scale in step 1
            align_centers: Whether to align centers in step 1  
            expansion_factor: Bounding box expansion factor for step 2
            file_format: Output file format for step 5
            
        Returns:
            List of saved file paths
        """
        print("🚀 Starting complete mesh segmentation pipeline...")
        print("=" * 60)
        
        # Step 1: Load and normalize
        self.step1_load_and_normalize_meshes(scale_normalize, align_centers)
        
        # Step 2: Identify bounding volumes
        self.step2_identify_part_bounding_volumes(expansion_factor=expansion_factor)
        
        # Step 3: Crop low-quality mesh
        self.step3_crop_low_quality_mesh()
        
        # Step 4: Refine segments
        self.step4_refine_mesh_segments()
        
        # Step 5: Save results
        saved_files = self.step5_save_separated_objects(file_format)
        
        print("=" * 60)
        print("🎉 Pipeline completed successfully!")
        print(f"📊 Results: {len(saved_files)} parts extracted")
        print(f"📁 Output: {self.output_dir}")
        
        return saved_files


# Example usage and testing
def main():
    """Example usage of the mesh segmentation pipeline."""
    
    # Example paths - replace with your actual file paths
    high_res_path = "path/to/high_quality_model.ply"  # Multi-part reference model
    low_res_path = "path/to/low_quality_model.ply"    # Single mesh to segment
    
    # Create pipeline instance
    pipeline = MeshSegmentationPipeline(
        high_res_path=high_res_path,
        low_res_path=low_res_path,
        output_dir="segmented_parts"
    )
    
    # Run complete pipeline
    try:
        saved_files = pipeline.run_complete_pipeline(
            scale_normalize=True,
            align_centers=True,
            expansion_factor=0.15,  # 15% expansion of bounding boxes
            file_format="ply"
        )
        
        print(f"\n🎯 Successfully segmented mesh into {len(saved_files)} parts:")
        for file_path in saved_files:
            print(f"  - {file_path}")
            
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()