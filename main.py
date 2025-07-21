from pipeline import MeshSegmentationPipeline

# Initialize pipeline with FBX files
pipeline = MeshSegmentationPipeline(
    high_res_path="input_models/highq.ply",
    low_res_path="input_models/lowq.ply",
    output_dir="segmented_parts"
)

# Run complete pipeline
saved_files = pipeline.run_complete_pipeline()
print(f"Created {len(saved_files)} parts from FBX files!")