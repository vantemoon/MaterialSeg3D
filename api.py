import os
import sys
import io
import time
import cv2
import json
import base64
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image
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

app = Flask(__name__)

# ------------------
# Helper Functions
# ------------------

def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=fmt)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def np_to_base64(np_img: np.ndarray, fmt: str = "PNG") -> str:
    im = Image.fromarray(np_img)
    buffered = io.BytesIO()
    im.save(buffered, format=fmt)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def extract_zip_file(zip_file) -> str:
    """
    Extracts the uploaded ZIP file to a temporary directory.
    Returns the path to the extracted folder.
    """
    tmp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)
    return tmp_dir

# ------------------
# Workflow Functions
# ------------------

def get_rendering(sample_folder: str):
    """
    Runs a Blender-based rendering on the .obj file in the sample_folder.
    Returns five rendered views as numpy arrays (RGB).
    """
    sample_folder = sample_folder.rstrip('/')
    sample = sample_folder.split('/')[-1]
    file_list = os.listdir(sample_folder)
    for file in file_list:
        if file.endswith('.obj'):
            BLENDER_PATH = "/opt/blender-2.90.0-linux64/blender"
            cmd = (f'{BLENDER_PATH} -b -P ./GET3D/render_shapenet_data/render_shapenet.py '
                   f'-- --output ./output {os.path.join(sample_folder, file)} '
                   f'--scale 1 --views 41 --resolution 1024 >> tmp.out')
            os.system(cmd)
            img_list = []
            for i in range(5):
                out_dir = os.path.join('./output/Image', sample, f'{sample}_{i}.png')
                img = cv2.imread(out_dir)
                if img is None:
                    print(f"Warning: Could not read image at {out_dir}")
                    continue
                # Convert BGR to RGB
                img_rgb = img[:, :, [2, 1, 0]]
                img_list.append(img_rgb)
            if len(img_list) < 5:
                raise Exception("Not enough rendered images were produced.")
            return img_list[0], img_list[1], img_list[2], img_list[3], img_list[4]
    raise Exception("No .obj file found in the sample folder.")

def get_segmentation(sample_folder: str, category: str):
    """
    Runs segmentation on the rendered images.
    Returns five segmentation visualization images as numpy arrays (RGB).
    """
    def getFileList(dir, Filelist, ext=None, skip=None, spec=None):
        newDir = dir
        if os.path.isfile(dir):
            if ext is None or dir.endswith(ext):
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
    img_list = getFileList(os.path.join('./output/Image/', sample), [], ext='png')
    for img in img_list:
        image = cv2.imread(img)
        back1 = np.array([0, 0, 0])
        back2 = np.array([1, 1, 1])
        target_color = np.array([255, 255, 255])
        image[np.all(image == back1, axis=2)] = target_color
        image[np.all(image == back2, axis=2)] = target_color
        save_file = img.replace('Image', 'Image_white')
        save_dir = os.path.dirname(save_file) + os.sep
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(save_file, image)
    seg_list = getFileList(save_dir, [], ext='png')
    config_path = os.path.join('/app/MaterialSeg3D/mmsegmentation/work_dir', category, f'3D_texture_{category}.py')
    checkpoint_path = os.path.join('/app/MaterialSeg3D/mmsegmentation/work_dir', category, 'ckpt.pth')
    model = init_model(config_path, checkpoint_path)
    print("Number of segmentation files:", len(seg_list))
    i = 0
    for img in seg_list:
        save_dir_local = os.path.dirname(img) + os.sep
        save_dir_pred = save_dir_local.replace('Image_white', 'predict')
        os.makedirs(save_dir_pred, exist_ok=True)
        img_name = os.path.basename(img)
        save_path = os.path.join(save_dir_pred, img_name)
        visual_dir = save_dir_local.replace('Image_white', 'vis')
        os.makedirs(visual_dir, exist_ok=True)
        visual_path = os.path.join(visual_dir, img_name)
        result = inference_model(model, img)
        predict = result.pred_sem_seg.data
        save_pred = np.squeeze(predict.cpu().numpy())
        save_mapping = transfer2(save_pred, mapping)
        cv2.imwrite(save_path, save_mapping)
        vis_img = to_rgb(save_pred, palette)
        cv2.imwrite(visual_path, vis_img)
        i += 1
    vis_list = []
    for i in range(5):
        seg_result = os.path.join('./output/vis', sample, f'{sample}_{i}.png')
        seg = cv2.imread(seg_result)
        if seg is None:
            raise Exception(f"Segmentation result not found: {seg_result}")
        seg_rgb = seg[:, :, [2, 1, 0]]
        vis_list.append(seg_rgb)
    if len(vis_list) < 5:
        raise Exception("Not enough segmentation images produced.")
    return vis_list[0], vis_list[1], vis_list[2], vis_list[3], vis_list[4]

def render_to_uv(sample_folder: str, category: str):
    """
    Converts rendered and segmented results into an ORM UV map.
    Returns the ORM image (RGB) as a numpy array.
    """
    sample_folder = sample_folder.rstrip('/')
    sample = sample_folder.split('/')[-1]
    original_dir = os.getcwd()
    os.chdir('./Text2Tex')
    cmd = (f'python ./scripts/view_2_UV.py --cuda 2 --work_dir ../output/predict '
           f'--sample_dir {sample_folder} --sample {sample} --img_size 512 --category {category}')
    os.system(cmd)
    os.chdir(original_dir)
    ORM_dir = os.path.join('./output/ORM/', sample, 'ORM.png')
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
        if file.endswith('png'):
            uv = Image.open(os.path.join(sample_folder, file))
            break
    BLENDER_PATH = "/opt/blender-2.90.0-linux64/blender"
    cmd = f'{BLENDER_PATH} -b -P trans_glb.py -- --obj_file {sample_folder}'
    os.system(cmd)
    mesh_path = os.path.join(sample_folder, f'{sample}_raw.glb')
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

@app.route('/display', methods=['POST'])
def display_endpoint():
    """
    Expects a multipart/form-data request with:
      - zip_file: the ZIP file containing OBJ, MTL, and PNG files.
    Returns a base64-encoded UV image and the generated mesh file path.
    """
    if 'zip_file' not in request.files:
        return jsonify({"error": "Missing zip_file parameter"}), 400

    zip_file = request.files['zip_file']
    sample_folder = extract_zip_file(zip_file)

    try:
        uv, mesh_path = display(sample_folder)
        uv_b64 = pil_to_base64(uv) if uv is not None else ""
        return jsonify({"uv_image": uv_b64, "mesh_path": mesh_path})
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("Display endpoint error:", error_details)
        return jsonify({"error": str(e)}), 500

@app.route('/get_rendering', methods=['POST'])
def get_rendering_endpoint():
    """
    Expects a multipart/form-data request with:
      - zip_file: the ZIP file containing the necessary asset files.
    Returns five rendered views as base64-encoded images.
    """
    if 'zip_file' not in request.files:
        return jsonify({"error": "Missing zip_file parameter"}), 400

    zip_file = request.files['zip_file']
    sample_folder = extract_zip_file(zip_file)

    try:
        imgs = get_rendering(sample_folder)  # Returns a tuple of 5 numpy arrays
        imgs_b64 = [np_to_base64(img) for img in imgs]
        return jsonify({
            "view1": imgs_b64[0],
            "view2": imgs_b64[1],
            "view3": imgs_b64[2],
            "view4": imgs_b64[3],
            "view5": imgs_b64[4],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_segmentation', methods=['POST'])
def get_segmentation_endpoint():
    """
    Expects a multipart/form-data request with:
      - zip_file: the ZIP file containing the necessary asset files.
      - category: the asset category (e.g., "car", "furniture").
    Returns five segmentation images as base64-encoded strings.
    """
    if 'zip_file' not in request.files:
        return jsonify({"error": "Missing zip_file parameter"}), 400

    zip_file = request.files['zip_file']
    sample_folder = extract_zip_file(zip_file)
    category = request.form.get("category")
    if not category:
        return jsonify({"error": "Missing category parameter"}), 400

    try:
        seg_imgs = get_segmentation(sample_folder, category)
        seg_imgs_b64 = [np_to_base64(img) for img in seg_imgs]
        return jsonify({
            "seg1": seg_imgs_b64[0],
            "seg2": seg_imgs_b64[1],
            "seg3": seg_imgs_b64[2],
            "seg4": seg_imgs_b64[3],
            "seg5": seg_imgs_b64[4],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/render_to_uv', methods=['POST'])
def render_to_uv_endpoint():
    """
    Expects a multipart/form-data request with:
      - zip_file: the ZIP file containing the necessary asset files.
      - category: the asset category.
    Returns the ORM UV map as a base64-encoded image.
    """
    if 'zip_file' not in request.files:
        return jsonify({"error": "Missing zip_file parameter"}), 400

    zip_file = request.files['zip_file']
    sample_folder = extract_zip_file(zip_file)
    category = request.form.get("category")
    if not category:
        return jsonify({"error": "Missing category parameter"}), 400

    try:
        orm_img = render_to_uv(sample_folder, category)
        orm_b64 = np_to_base64(orm_img)
        return jsonify({"ORM_image": orm_b64})
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
