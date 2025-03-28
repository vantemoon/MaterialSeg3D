import os
import sys
import io
import re
import time
import cv2
import json
import base64
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
import subprocess
import tempfile
import zipfile

sys.path.append("./mmsegmentation")
sys.path.append("./Text2Tex")
sys.path.append("./GET3D")

# Import mmsegmentation APIs and other libraries
from mmseg.apis import init_model, inference_model
import torch
from tqdm import tqdm
from scipy import stats

# Import pytorch3d and our helper modules from the material pipeline
from pytorch3d.renderer import TexturesUV, AmbientLights, SoftPhongShader
from pytorch3d.ops import interpolate_face_attributes
from lib.render_helper import init_renderer, render
from lib.projection_helper import (build_backproject_mask, build_diffusion_mask,
                                     compose_quad_mask, compute_view_heat)
from lib.shading_helper import init_soft_phong_shader
from lib.camera_helper import init_camera, init_viewpoints
from lib.shading_helper import BlendParams, init_soft_phong_shader, init_flat_texel_shader
from lib.vis_helper import (visualize_outputs, visualize_quad_mask,
                            visualize_principle_viewpoints, visualize_refinement_viewpoints)
from lib.diffusion_helper import (get_controlnet_depth, get_inpainting,
                                  apply_controlnet_depth, apply_inpainting_postprocess)
from lib.projection_helper import get_all_4_locations, select_viewpoint
from lib.mesh_helper import init_mesh_2, apply_offsets_to_mesh, adjust_uv_map
from lib.io_helper import save_backproject_obj, save_args, save_viewpoints

# Mapping and palette for segmentation post-processing
mapping = {3:2, 4:2, 5:3, 6:4, 7:5, 8:6, 9:6, 10:7, 11:7,
           12:8, 13:9, 14:10, 15:11, 16:12, 17:13, 18:14, 19:15}
palette = [[0, 0, 0], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180],
           [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
           [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32], [100, 100, 0]]

app = Flask(__name__, static_url_path='/static', static_folder='/shared')
CORS(app)

# ------------------
# Helper Functions
# ------------------

def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=fmt)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{encoded}"

def np_to_base64(np_img: np.ndarray, fmt: str = "PNG") -> str:
    pil_img = Image.fromarray(np_img.astype(np.uint8))
    buffered = io.BytesIO()
    pil_img.save(buffered, format=fmt)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{encoded}"

def extract_zip_file(zip_file) -> str:
    """
    Extracts the uploaded ZIP file to a temporary directory.
    Returns the path to the extracted folder.
    """
    tmp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)
    return tmp_dir

def natural_key(string):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', string)]

# ------------------
# Workflow Functions
# ------------------

def get_rendering(sample_folder: str):
    """
    Runs a Blender-based rendering on the .obj file in the sample_folder.
    Returns five rendered views as numpy arrays (RGB).
    Assumes that the rendered images are stored under:
      /shared/output/Image/<sample>/<any PNG files>
    """
    sample_folder = sample_folder.rstrip('/')
    sample = sample_folder.split('/')[-1]
    file_list = os.listdir(sample_folder)
    target_obj = None
    for file in file_list:
        if file.endswith('.obj'):
            target_obj = os.path.join(sample_folder, file)
            break
    if not target_obj:
        raise Exception("No .obj file found in the sample folder.")

    BLENDER_PATH = "/opt/blender-2.90.0-linux64/blender"
    # Use the shared folder for output.
    render_folder = "/shared/output/"
    os.makedirs(render_folder, exist_ok=True)

    blender_script = "/app/MaterialSeg3D/GET3D/render_shapenet_data/render_shapenet.py"
    # Note: The '--' separator ensures the arguments are passed to the script.
    cmd = (f'{BLENDER_PATH} -b -P {blender_script} '
           f'-- --output_folder {render_folder} {target_obj} '
           f'--scale 1 --views 41 --resolution 1024 >> /shared/tmp.out')
    print("Running rendering command:", cmd)
    ret = os.system(cmd)
    print("Rendering command returned:", ret)

    # Look for images in the expected subfolder.
    image_dir = os.path.join(render_folder, 'Image', sample)
    if not os.path.exists(image_dir):
        print("No 'Image' subfolder found; using render_folder as image directory.")
        image_dir = render_folder
    else:
        # Even if the folder exists, check that it contains PNG files.
        if len([f for f in os.listdir(image_dir) if f.lower().endswith('.png')]) == 0:
            print("'Image' subfolder is empty; using render_folder as image directory.")
            image_dir = render_folder
        else:
            print("'Image' subfolder found and not empty; using it as image directory.")

    png_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith('.png')]
    png_files.sort(key=natural_key)
    print("Found PNG files:", png_files)

    if len(png_files) < 5:
        raise Exception("Not enough rendered images were produced. Found: " + str(len(png_files)))

    img_list = []
    for f in png_files[:5]:
        print("Loading image:", f)
        img = cv2.imread(f)
        if img is None:
            print("Warning: Failed to read image", f)
        else:
            # Convert from BGR to RGB.
            img_rgb = img[:, :, [2, 1, 0]]
            img_list.append(img_rgb)
    if len(img_list) < 5:
        raise Exception("Not enough rendered images were produced after reading files.")
    
    return tuple(img_list)

def get_segmentation(sample_folder: str, category: str):
    """
    Runs segmentation on the rendered images.
    Returns five segmentation visualization images as numpy arrays (RGB).
    This function:
      1. Recursively collects PNG files from the rendered images stored at:
         /shared/output/Image/<sample>/
      2. Processes them (replacing black/white backgrounds) and saves them to:
         /shared/output/Image_white/<sample>/
      3. Runs the segmentation model on these processed images, writing outputs to:
         /shared/output/predict and then visualizations to /shared/output/vis/<sample>/
      4. Loads and returns the first five visualization images.
    """
    # Helper recursive file search function.
    def getFileList(dir, Filelist, ext=None, skip=None, spec=None):
        newDir = dir
        if os.path.isfile(dir):
            if ext is None:
                Filelist.append(dir)
            else:
                if ext in dir[-3:]:
                    Filelist.append(dir)
        elif os.path.isdir(dir):
            for s in os.listdir(dir):
                if os.path.isdir(os.path.join(dir, s)):
                    newDir = os.path.join(dir, s)
                    getFileList(newDir, Filelist, ext, skip, spec)
                else:
                    acpt = True
                    if skip is not None:
                        for skipi in skip:
                            if skipi in s:
                                acpt = False
                                break
                    if not acpt:
                        continue
                    else:
                        sp = False
                        if spec is not None:
                            for speci in spec:
                                if speci in s:
                                    sp = True
                                    break
                        else:
                            sp = True
                        if not sp:
                            continue
                        else:
                            newDir = os.path.join(dir, s)
                            getFileList(newDir, Filelist, ext, skip, spec)
        return Filelist

    def to_rgb(label, palette):
        h, w = label.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                value = label[i, j]
                rgb[i, j] = palette[value]
        return rgb

    def transfer2(image, mapping):
        image_base = image.copy()
        for i in mapping.keys():
            image[image_base == i] = mapping[i]
        return image

    sample_folder = sample_folder.rstrip('/')
    sample = sample_folder.split('/')[-1]

    # Find rendered images from the first model in /shared/output/Image/<sample>
    rendered_dir = os.path.join('/shared/output/Image', sample)
    img_list = getFileList(rendered_dir, [], ext='png')
    print("Getting images from:", rendered_dir)
    
    # Process these images and save into /shared/output/Image_white/<sample>
    save_dir = os.path.join('/shared/output/Image_white', sample)
    os.makedirs(save_dir, exist_ok=True)
    for img in img_list:
        image = cv2.imread(img)
        if image is None:
            continue
        back1 = np.array([0, 0, 0])
        back2 = np.array([1, 1, 1])
        target_color = np.array([255, 255, 255])
        image[np.all(image == back1, axis=2)] = target_color
        image[np.all(image == back2, axis=2)] = target_color
        # Replace "Image" with "Image_white" in the path.
        save_file = img.replace('/Image/', '/Image_white/')
        save_dir_local = os.path.dirname(save_file) + os.sep
        os.makedirs(save_dir_local, exist_ok=True)
        print("Saving processed image to:", save_file)
        cv2.imwrite(save_file, image)

    # Use the directory where processed images were saved.
    seg_list = getFileList(save_dir, [], ext='png')

    # Initialize the segmentation model.
    config_path = os.path.join('/app/MaterialSeg3D/mmsegmentation/work_dir', category, f'3D_texture_{category}.py')
    checkpoint_path = os.path.join('/app/MaterialSeg3D/mmsegmentation/work_dir', category, 'ckpt.pth')
    model = init_model(config_path, checkpoint_path)

    print("Number of processed images:", len(seg_list))
    i = 0
    # For each processed image, run segmentation and save outputs.
    for img in seg_list:
        img_name = os.path.basename(img)
        save_dir_local = os.path.dirname(img) + os.sep
        # Replace "Image_white" with "predict" for intermediate segmentation outputs.
        save_dir_predict = save_dir_local.replace('Image_white', 'predict')
        print("Saving segmentation prediction to:", save_dir_predict)
        os.makedirs(save_dir_predict, exist_ok=True)
        save_path = os.path.join(save_dir_predict, img_name)

        # Also define a directory for visualizations (segmentation overlays).
        visual_dir = save_dir_predict.replace('predict', 'vis')
        print("Saving segmentation visualization to:", visual_dir)
        visual_path = save_path.replace('predict', 'vis')
        os.makedirs(visual_dir, exist_ok=True)

        result = inference_model(model, img)
        predict = result.pred_sem_seg.data
        save_pred = np.squeeze(predict.cpu().numpy())
        # 'mapping' should be defined globally or imported.
        save_mapping = transfer2(save_pred, mapping)
        cv2.imwrite(save_path, save_mapping)

        vis_img = to_rgb(save_pred, palette)
        cv2.imwrite(visual_path, vis_img)
        i += 1

    # Now load segmentation visualization images from /shared/output/vis/<sample>
    vis_dir = os.path.join('/shared/output/vis', sample)
    if not os.path.exists(vis_dir):
        raise Exception(f"Segmentation directory not found: {vis_dir}")
    files_in_vis = os.listdir(vis_dir)
    print("Contents of segmentation directory:", files_in_vis)
    vis_files = [os.path.join(vis_dir, f) for f in files_in_vis if f.lower().endswith('.png')]
    vis_files.sort(key=natural_key)
    
    if len(vis_files) < 5:
        raise Exception("Not enough segmentation images produced. Found: " + str(len(vis_files)))
    
    vis_list = []
    for f in vis_files[:5]:
        seg = cv2.imread(f)
        if seg is None:
            raise Exception(f"Segmentation result not found or could not be read: {f}")
        seg_rgb = seg[:, :, [2, 1, 0]]
        vis_list.append(seg_rgb)
    
    return tuple(vis_list)

def render_to_uv(sample_folder: str, category: str):
    """
    Converts rendered and segmented results into an ORM UV map.
    Returns the ORM image (RGB) as a numpy array.
    """
    sample_folder = sample_folder.rstrip('/')
    sample = sample_folder.split('/')[-1]
    original_dir = os.getcwd()

    os.chdir('/app/MaterialSeg3D/Text2Tex')
    work_dir = "/shared/output/predict"
    os.makedirs(work_dir, exist_ok=True)

    # Construct command using absolute paths:
    # --work_dir now uses the absolute path, and --sample_dir is the sample folder.
    cmd = (
        f'python ./scripts/view_2_UV_inference.py --cuda 2 --work_dir {work_dir} '
        f'--sample_dir {sample_folder} --sample {sample} --img_size 512 --category {category}'
    )
    print("Running render_to_uv command:", cmd)
    os.system(cmd)
    os.chdir(original_dir)

    ORM_dir = os.path.join("/shared/output/ORM", sample, "ORM.png")
    os.system('cp ' + ORM_dir + ' ' + sample_folder)
    ORM = cv2.imread(ORM_dir)
    if ORM is None:
        raise Exception("ORM image not found.")
    ORM_rgb = ORM[:, :, [2, 1, 0]]
    return ORM_rgb

def display(sample_folder: str):
    """
    Runs a display step that launches Blender to convert the object into a mesh.
    Returns a UV image (from one of the PNG files in the folder) and the path to the generated mesh.
    """
    sample_folder = sample_folder.rstrip('/')
    sample = sample_folder.split('/')[-1]
    uv = None
    for file in os.listdir(sample_folder):
        if file.endswith('.png'):
            uv = Image.open(os.path.join(sample_folder, file))
            break
    BLENDER_PATH = "/opt/blender-2.90.0-linux64/blender"
    cmd = [BLENDER_PATH, "-b", "-P", "trans_glb.py", "--", "--obj_file", sample_folder]
    print("Running Blender command:", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Blender stdout:", result.stdout)
        print("Blender stderr:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Blender command failed with code", e.returncode)
        print("Error output:", e.stderr)
        raise Exception("Blender command failed; see logs for details.")
    
    mesh_path = f"/shared/{sample}/mesh.obj"
    if not os.path.exists(mesh_path):
        raise Exception(f"Mesh file '{mesh_path}' not found.")
    
    return uv, mesh_path

def example():
    """
    Returns example images for the material generation step.
    Loads images from disk and returns them as base64 strings.
    """
    paths = ['./figure/material_ue.png', './figure/material_car.png',
             './figure/raw_ue.png', './figure/raw_car.png']
    imgs = []
    for path in paths:
        if not os.path.exists(path):
            imgs.append("")
        else:
            img = Image.open(path)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            imgs.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    return imgs[0], imgs[1], imgs[2], imgs[3]

# ------------------
# API Endpoints
# ------------------

@app.route('/shared/<path:filename>')
def shared_files(filename):
    return send_from_directory('/shared', filename)

@app.route('/display', methods=['POST'])
def display_endpoint():
    print("Display endpoint called.")
    data = request.get_json()
    sample_folder = data.get("sample_folder")
    if not sample_folder:
        return jsonify({"error": "Missing sample_folder parameter"}), 400
    try:
        if not os.path.exists(sample_folder):
            raise Exception(f"Sample folder '{sample_folder}' does not exist.")
        print("Display endpoint: sample_folder =", sample_folder)
        print("Files in sample_folder:", os.listdir(sample_folder))
        sample = sample_folder.rstrip('/').split('/')[-1]
        texture_file = f"http://localhost:8080/static/{sample}/texture.png"
        uv = Image.open(texture_file)
        mesh_path = f"http://localhost:8080/static/{sample}/mesh.obj"
        uv_b64 = pil_to_base64(uv) if uv is not None else ""
        return jsonify({"uv_image": uv_b64, "mesh_path": mesh_path})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/get_rendering', methods=['POST'])
def get_rendering_endpoint():
    if 'zip_file' in request.files:
        zip_file = request.files['zip_file']
        sample_folder = extract_zip_file(zip_file)
    else:
        data = request.get_json(silent=True)
        sample_folder = data.get("zip_file") if data else None
        if not sample_folder:
            return jsonify({"error": "Missing zip_file parameter"}), 400

    try:
        rendered_images = get_rendering(sample_folder)
        sample = sample_folder.rstrip('/').split('/')[-1]

        urls = []
        for i in range(5):
            url = f"http://localhost:8080/static/output/vis/{sample}/{sample}_{i}.png"
            urls.append(url)
        return jsonify({
            "view1": urls[0],
            "view2": urls[1],
            "view3": urls[2],
            "view4": urls[3],
            "view5": urls[4],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_segmentation', methods=['POST'])
def get_segmentation_endpoint():
    data = request.get_json(force=True)
    sample_folder = data.get("zip_file")
    category = data.get("category")
    if not sample_folder or not category:
        return jsonify({"error": "Missing zip_file or category parameter"}), 400

    try:
        seg_imgs = get_segmentation(sample_folder, category)
        sample = sample_folder.rstrip('/').split('/')[-1]
        urls = []
        for i in range(5):
            url = f"http://localhost:8080/static/output/vis/{sample}/{sample}_{i}.png"
            urls.append(url)
        return jsonify({
            "seg1": urls[0],
            "seg2": urls[1],
            "seg3": urls[2],
            "seg4": urls[3],
            "seg5": urls[4],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/render_to_uv', methods=['POST'])
def render_to_uv_endpoint():
    """
    Generate the ORM UV map.
    Expects a JSON payload with:
      - zip_file: The folder path containing the generated outputs.
      - category: The asset category.
    Returns the ORM UV map as a base64-encoded image.
    """
    # First, check for file upload (if any)
    if 'zip_file' in request.files:
        zip_file = request.files['zip_file']
        sample_folder = extract_zip_file(zip_file)
        category = request.form.get("category")
    else:
        # Otherwise, parse JSON payload.
        data = request.get_json(silent=True)
        sample_folder = data.get("zip_file") if data else None
        category = data.get("category") if data else None
        if not sample_folder or not category:
            return jsonify({"error": "Missing zip_file or category parameter"}), 400

    try:
        orm_img = render_to_uv(sample_folder, category)
        sample = sample_folder.rstrip('/').split('/')[-1]
        orm_dir = os.path.join("/shared/output/ORM", sample)
        os.makedirs(orm_dir, exist_ok=True)
        orm_file_path = os.path.join(orm_dir, "ORM.png")
        cv2.imwrite(orm_file_path, orm_img)

        os.system('cp ' + orm_file_path + ' ' + sample_folder)

        orm_url = f"http://localhost:8080/static/output/ORM/{sample}/ORM.png"
        return jsonify({"ORM_image_url": orm_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/example', methods=['GET'])
def example_endpoint():
    """
    Returns example images for material generation.
    """
    try:
        mat_ue, mat_car, raw_ue, raw_car = example()
        return jsonify({
            "mat_ue": mat_ue,
            "mat_car": mat_car,
            "raw_ue": raw_ue,
            "raw_car": raw_car,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download_material', methods=['GET'])
def download_material_endpoint():
    """
    Download a zip file containing:
      - the white model (OBJ file),
      - the albedo texture (PNG file),
      - the ORM map (PNG file)
    Expects a query parameter:
      - sample_folder: the folder containing these files.
    """
    sample_folder =  request.args.get("sample_folder")
    if not sample_folder or not os.path.exists(sample_folder):
        return jsonify({"error": "Sample folder not found"}), 404

    # Look for files: assume .obj file is the white model,
    # the albedo texture is the first PNG file without "ORM" in its name,
    # and the ORM map is a PNG file with "ORM" in its name.
    obj_file = None
    albedo_file = None
    orm_file = None
    for f in os.listdir(sample_folder):
        if f.endswith('.obj'):
            obj_file = os.path.join(sample_folder, f)
        elif f.endswith('.png'):
            if "ORM" in f:
                orm_file = os.path.join(sample_folder, f)
            else:
                if not albedo_file:
                    albedo_file = os.path.join(sample_folder, f)
    if not obj_file or not albedo_file or not orm_file:
        return jsonify({"error": "Required files not found"}), 404

    zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(obj_file, arcname=os.path.basename(obj_file))
        zipf.write(albedo_file, arcname=os.path.basename(albedo_file))
        zipf.write(orm_file, arcname=os.path.basename(orm_file))
    return send_file(zip_path, as_attachment=True, download_name="material.zip", mimetype="application/zip")

# ------------------
# Main
# ------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
