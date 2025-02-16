#!/usr/bin/env python3
import argparse
import os
from PIL import Image

# Import the relevant functions from gradio_demo.py
# Adjust the path as needed, or place them in the same folder.
from gradio_demo import (
    display,
    get_rendering,
    get_segmentation,
    render_to_uv,
    example
)

def run_matseg(input_dir, category):
    """
    Runs the typical matseg pipeline:
      1) display() -> returns albedo_uv, input_mesh
      2) get_rendering() -> returns multiple viewpoint images
      3) get_segmentation() -> returns viewpoint segmentations
      4) render_to_uv() -> produces ORM.png

    Modify as needed if you don't want all these steps or want to store outputs differently.
    """

    # Step 1: "display" returns (albedo_uv, mesh_path)
    #   - In gradio_demo, this loads the .obj and .png from input_dir
    albedo_uv, mesh_path = display(input_dir)
    print(f"[matseg_run] display() => albedo_uv={albedo_uv}, mesh_path={mesh_path}")

    # Step 2: "get_rendering" => multiple viewpoint images
    v1, v2, v3, v4, v5 = get_rendering(input_dir)
    print("[matseg_run] get_rendering() => 5 viewpoint images rendered")

    # Step 3: "get_segmentation" => returns segmentations
    seg1, seg2, seg3, seg4, seg5 = get_segmentation(input_dir, category)
    print("[matseg_run] get_segmentation() => 5 segmentations")

    # Step 4: "render_to_uv" => produces the final ORM UV map
    orm_rgb = render_to_uv(input_dir, category)
    print("[matseg_run] render_to_uv() => produced ORM (RGB array)")

    # We also expect "ORM.png" to appear in the input_dir after the pipeline.
    # You could do more here, like rename or move final outputs.
    print("[matseg_run] Done. Check your folder for final outputs (ORM.png, etc.)")


def main():
    parser = argparse.ArgumentParser(description="Run material segmentation (MatSeg) headlessly.")
    parser.add_argument("--input-dir", required=True, help="Path to the folder containing .obj, .png, etc.")
    parser.add_argument("--category", required=True, choices=["car","furniture","building","instrument","plant"],
                        help="Object category (affects the trained model used).")

    args = parser.parse_args()

    # Ensure path is absolute or handle relative carefully
    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise ValueError(f"Invalid directory: {input_dir}")

    run_matseg(input_dir, args.category)

if __name__ == "__main__":
    main()
