import os
import imageio
import uuid
import open3d as o3d
import numpy as np
# from IPython.display import Image as ImageDisplay
# 加入了尺寸查看代码get_ply_scale
from inference import Inference, ready_gaussian_for_video_rendering, render_video, load_image, load_single_mask, display_image, make_scene, interactive_visualizer, get_ply_scale
# from notebook.return_mesh_to_realscale import normalize_model
PATH = os.getcwd()
TAG = "hf"
# 加入相对读取图片代码，归一化到data_online
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
config_path = os.path.join(PROJECT_ROOT, "sam-3d-objects", "checkpoints", TAG, "pipeline.yaml")
IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "data_online", "2", "preprocess_image", "00000_rgb.png"
)
IMAGE_NAME = os.path.basename(os.path.dirname(IMAGE_PATH))
###############################################################
# config_path = f"{PATH}/../checkpoints/{TAG}/pipeline.yaml"
#config_path = os.path.join(PATH, "checkpoints", TAG, "pipeline.yaml") # ⬅️ 修正后的路径计算
inference = Inference(config_path, compile=False)
# IMAGE_PATH = f"{PATH}/notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png"# /data/mmw/test/sam-3d-objects/notebook/images/cup
# IMAGE_PATH = f"{PATH}/../data_online/1/preprocess_images/airpods.jpg"
# IMAGE_NAME = os.path.basename(os.path.dirname(IMAGE_PATH))
image = load_image(IMAGE_PATH)
mask = load_single_mask(os.path.dirname(IMAGE_PATH), index=1)
display_image(image, masks=[mask])
# run model
output = inference(image, mask, seed=42)

# export gaussian splat (as point cloud)
# output["gs"].save_ply(f"{PATH}/notebook/gaussians/single/{IMAGE_NAME}.ply")# export mesh with texture (使用 to_glb 函数)
if "mesh" in output and len(output["mesh"]) > 0:
    from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import to_glb
    
    mesh_extract_result = output["mesh"][0]  # MeshExtractResult 对象
    gaussian = output["gs"]  # Gaussian 对象用于纹理烘焙
    
    print("正在生成带贴图的 mesh...")
    print("  - 网格后处理: 简化 + 填充孔洞")
    print("  - 颜色来源: 顶点颜色（从原始mesh）")
    print("  - 贴图分辨率: 1024x1024")
    
    # 使用 to_glb 生成带顶点颜色的 mesh
    mesh_trimesh = to_glb(
        app_rep=gaussian,
        mesh=mesh_extract_result,
        simplify=0.95,              # 简化到原始面片的 5%
        fill_holes=True,            # 填充孔洞
        fill_holes_max_size=0.04,   # 最大孔洞尺寸
        texture_size=1024,          # 贴图分辨率（顶点颜色模式不使用）
        debug=False,
        verbose=True,
        with_mesh_postprocess=True,    # 启用网格后处理
        with_texture_baking=False,     # 禁用纹理烘焙（使用顶点颜色）
        use_vertex_color=True,         # 使用顶点颜色
        rendering_engine="nvdiffrast"
    )
    
    # 导出带贴图的 GLB 和 OBJ
    mesh_dir = f"{PATH}/notebook/meshes/single"
    os.makedirs(mesh_dir, exist_ok=True)
    
    glb_path = f"{mesh_dir}/{IMAGE_NAME}_textured.glb"
    obj_path = f"{mesh_dir}/{IMAGE_NAME}_textured.obj"
    
    # 导出 GLB (推荐，包含贴图)
    mesh_trimesh.export(glb_path)
    
    # 导出 OBJ + MTL + PNG (贴图会自动保存为 .png)
    mesh_trimesh.export(obj_path)
    
    print(f"✓ 带贴图的 Mesh 已导出:")
    print(f"  - GLB: {glb_path}")
    print(f"  - OBJ: {obj_path}")
    print(f"  - 顶点数: {len(mesh_trimesh.vertices)}")
    print(f"  - 面片数: {len(mesh_trimesh.faces)}")
    if hasattr(mesh_trimesh.visual, 'material') and mesh_trimesh.visual.material:
        print(f"  - 贴图: ✓ (1024x1024)")
# output["mesh"].save_(f"{PATH}/notebook/gaussians/single/{IMAGE_NAME}_mesh.ply")
PLY_PATH = f"{PATH}/notebook/gaussians/single/{IMAGE_NAME}.ply"
# render gaussian splat
scene_gs = make_scene(output)
scene_gs = ready_gaussian_for_video_rendering(scene_gs)

video = render_video(
    scene_gs,
    r=1,
    fov=60,
    pitch_deg=15,
    yaw_start_deg=-45,
    resolution=512,
)["color"]

# save video as gif
imageio.mimsave(
    os.path.join(f"{PATH}/notebook/gaussians/single/{IMAGE_NAME}.gif"),
    video,
    format="GIF",
    duration=1000 / 30,  # default assuming 30fps from the input MP4
    loop=0,  # 0 means loop indefinitely
)
get_ply_scale(PLY_PATH)
# notebook display
# ImageDisplay(url=f"gaussians/single/{IMAGE_NAME}.gif?cache_invalidator={uuid.uuid4()}")
# might take a while to load (black screen)
# interactive_visualizer(f"{PATH}/gaussians/single/{IMAGE_NAME}.ply")