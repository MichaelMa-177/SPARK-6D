"""Prepare one BOP RGB-D click input without loading scene poses or GT masks.

This adapter changes storage/units only. The existing product entry is called
separately with one frame, explicit --click, --depth-scale and --symmetries.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData,PlyElement,PlyListProperty


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def identity(value):
    value=np.ascontiguousarray(value)
    return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.tobytes()).hexdigest())


def _require(condition,message):
    if not condition:raise ValueError(message)


def _write_json(path,value):
    with Path(path).open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def verify_source(binding):
    path=Path(binding['path'])
    _require(str(path.resolve(strict=True))==binding['resolved_path']and path.stat().st_size==binding['bytes']
        and sha256(path)==binding['sha256'],'original source binding differs: '+str(path))
    return path


def original_model_info_metres(model_info_mm):
    """Scale original-frame translations/offsets; never pre-center symmetries."""
    _require(isinstance(model_info_mm,dict),'object models_info entry required')
    allowed={'diameter','min_x','min_y','min_z','size_x','size_y','size_z','symmetries_discrete','symmetries_continuous'}
    _require(set(model_info_mm)<=allowed and 'diameter'in model_info_mm,'unsupported BOP model metadata fields')
    metric=deepcopy(model_info_mm)
    for key in allowed-{'symmetries_discrete','symmetries_continuous'}:
        if key in metric:
            _require(type(metric[key])in (int,float)and np.isfinite(metric[key]),'finite millimetre metadata required')
            metric[key]=float(metric[key])*.001
    _require(metric['diameter']>0,'positive model diameter required')
    for original,target in zip(model_info_mm.get('symmetries_discrete',[]),metric.get('symmetries_discrete',[]),strict=True):
        value=np.asarray(original,dtype=np.float64)
        _require(value.shape==(16,)and np.isfinite(value).all(),'finite original 16-value BOP symmetry required')
        matrix=value.reshape(4,4).copy();_require(np.array_equal(matrix[3],[0,0,0,1]),'homogeneous symmetry required')
        matrix[:3,3]*=.001;target[:]=matrix.ravel().tolist()
    for original,target in zip(model_info_mm.get('symmetries_continuous',[]),metric.get('symmetries_continuous',[]),strict=True):
        _require(set(original)=={'axis','offset'},'explicit continuous axis and offset required')
        axis=np.asarray(original['axis'],dtype=np.float64);offset=np.asarray(original['offset'],dtype=np.float64)
        _require(axis.shape==offset.shape==(3,)and np.isfinite(axis).all()and np.isfinite(offset).all()
            and np.linalg.norm(axis)>0,'finite nonzero continuous axis/finite offset required')
        target['offset']=(offset*.001).tolist()  # Axis stays dimensionless, in its original order.
    return metric


def approved_camera(whitelist,*,scene_id,image_id,expected_K,expected_depth_scale_m):
    _require(whitelist.get('extrinsic_values_parsed')is False and whitelist.get('unknown_value_fields_never_parsed')is True,
        'camera must come from the reviewed whitelist without parsed extrinsics')
    matches=[entry['approved']for scene in whitelist['cameras']if scene['scene_id']==scene_id
        for entry in scene['entries']if entry['image_id']==image_id]
    _require(len(matches)==1 and set(matches[0])=={'cam_K','depth_scale'},'exact per-image approved camera required')
    value=matches[0];flat=np.asarray(value['cam_K'],dtype=np.float64)
    _require(flat.shape==(9,)and np.isfinite(flat).all(),'finite 9-value per-image camera required')
    K=flat.reshape(3,3);expected=np.asarray(expected_K,dtype=np.float64)
    _require(expected.shape==(3,3)and np.array_equal(K,expected),'per-image camera differs from sealed selected slot')
    _require(K[0,0]>0 and K[1,1]>0 and np.array_equal(K[2],[0,0,1]),'valid pinhole intrinsics required')
    scale=float(value['depth_scale'])*.001
    _require(np.isfinite(scale)and scale>0 and scale==expected_depth_scale_m,'raw depth unit differs from selected slot')
    return K.copy(),scale


def convert_original_cad(binding,destination,*,source_units):
    """Independent PLY with exact declared numeric vertices * .001 and row order.

    Preserve all other scalar/list properties and their declared types. Vertex
    xyz becomes float64 so export cannot introduce an extra float32 rounding.
    Only original pinned millimetre assets may enter this conversion.
    """
    _require(source_units=='bop_original_millimetres','reject already-converted or unspecified CAD units')
    source=verify_source(binding);destination=Path(destination)
    _require(source.resolve()!=destination.resolve()and not destination.exists(),'fresh independent metric CAD required')
    original=PlyData.read(str(source));_require(set(e.name for e in original.elements)=={'vertex','face'},'BOP vertex/face PLY required')
    vertices=original['vertex'].data;faces=original['face'].data
    _require({'x','y','z'}<=set(vertices.dtype.names),'explicit vertex xyz required')
    xyz=np.column_stack([vertices[k]for k in ('x','y','z')]).astype(np.float64)
    _require(np.isfinite(xyz).all()and len(xyz)>0,'finite nonempty original CAD required')
    _require('vertex_indices'in faces.dtype.names,'explicit face indices required')
    triangles=np.asarray([np.asarray(x)for x in faces['vertex_indices']])
    _require(triangles.ndim==2 and triangles.shape[1]==3 and triangles.dtype.kind in 'iu'
        and triangles.min()>=0 and triangles.max()<len(vertices),'finite indexed triangle topology required; no triangulation')
    dtype=[(name,'f8'if name in ('x','y','z')else vertices.dtype.fields[name][0])for name in vertices.dtype.names]
    converted=np.empty(len(vertices),dtype=dtype)
    for name in vertices.dtype.names:converted[name]=vertices[name].astype(np.float64)*.001 if name in ('x','y','z')else vertices[name]
    elements=[]
    for element in original.elements:
        data=converted if element.name=='vertex'else element.data.copy()
        lists={p.name:p for p in element.properties if isinstance(p,PlyListProperty)}
        elements.append(PlyElement.describe(data,element.name,len_types={k:v.len_dtype for k,v in lists.items()},
            val_types={k:v.val_dtype for k,v in lists.items()},comments=list(element.comments)))
    with destination.open('xb')as f:PlyData(elements,text=False,byte_order='<',comments=list(original.comments),obj_info=list(original.obj_info)).write(f)
    saved=PlyData.read(str(destination));metric=np.column_stack([saved['vertex'].data[k]for k in ('x','y','z')])
    _require(metric.dtype==np.float64 and np.array_equal(metric,xyz*.001),'metric CAD conversion changed coordinate values/order')
    for old,new in zip(original.elements,saved.elements,strict=True):
        _require(old.name==new.name and len(old.data)==len(new.data)and old.data.dtype.names==new.data.dtype.names,'PLY element/property row order changed')
        for name in old.data.dtype.names:
            if old.name=='vertex'and name in ('x','y','z'):continue
            a,b=old.data[name],new.data[name]
            _require(a.dtype==b.dtype and all(np.array_equal(x,y)for x,y in zip(a,b,strict=True)),'unitless property or topology changed')
    _require(sha256(source)==binding['sha256'],'source CAD changed during conversion')
    return dict(source=dict(binding),destination=str(destination),destination_sha256=sha256(destination),
        source_units=source_units,output_units='metres',conversion_factor=.001,conversions_applied=1,
        source_xyz_mm=identity(xyz),output_xyz_m=identity(metric),faces=identity(triangles),vertices=len(vertices),faces_count=len(faces),
        vertex_face_property_and_row_order_preserved=True,source_file_unchanged=True)


def product_loader_relation(path):
    """CPU evidence for unchanged public loader's existing process=True behavior."""
    import trimesh
    prepared=trimesh.load(str(path),force='mesh',process=False)
    actual=trimesh.load(str(path),force='mesh')  # Exact current public entry call.
    _require(isinstance(prepared,trimesh.Trimesh)and isinstance(actual,trimesh.Trimesh),'single product mesh required')
    coordinates={tuple(row)for row in np.asarray(prepared.vertices)}
    _require(all(tuple(row)in coordinates for row in np.asarray(actual.vertices)),'default loader introduced coordinates or changed units')
    _require(np.array_equal(prepared.bounds,actual.bounds),'default loader changed coordinate bounds')
    center=(actual.vertices.min(axis=0)+actual.vertices.max(axis=0))/2
    C=np.eye(4);C[:3,3]=-center
    return dict(prepared_vertices=len(prepared.vertices),product_default_vertices=len(actual.vertices),
        prepared_faces=len(prepared.faces),product_default_faces=len(actual.faces),
        product_vertex_order_exact=np.array_equal(prepared.vertices,actual.vertices),product_face_order_exact=np.array_equal(prepared.faces,actual.faces),
        all_product_vertices_are_prepared_coordinates=True,coordinate_bounds_exact=True,
        default_process_behavior_unchanged=True,runtime_order_preservation_claimed=False,
        model_center_m=center.tolist(),centered_from_original_m=C.tolist(),
        product_vertices=identity(np.asarray(actual.vertices)),product_faces=identity(np.asarray(actual.faces)))


def check_centered_symmetry(metric_info,model_center_m,actual_C=None):
    """Use the same product constructor; independently verify its C conjugation."""
    from pose_validity import SymmetrySpec
    center=np.asarray(model_center_m,dtype=np.float64);_require(center.shape==(3,)and np.isfinite(center).all(),'actual metric center required')
    C=np.eye(4);C[:3,3]=-center
    if actual_C is not None:
        raw=np.asarray(actual_C);_require(raw.shape==(4,4)and np.isfinite(raw).all()and np.allclose(raw,C,rtol=0,atol=2e-7),'actual FP C differs from actual center')
    spec=SymmetrySpec.from_bop(metric_info,model_center=center,unit_scale=1.)
    # SymmetrySpec inserts identity as a group member when absent.
    for raw in metric_info.get('symmetries_discrete',[]):
        original=np.asarray(raw,dtype=np.float64).reshape(4,4);expected=C@original@np.linalg.inv(C)
        _require(any(np.allclose(expected,value,rtol=0,atol=1e-12)for value in spec.discrete),'centered discrete symmetry fails C conjugation')
    for raw,(axis,offset)in zip(metric_info.get('symmetries_continuous',[]),spec.continuous,strict=True):
        original_axis=np.asarray(raw['axis'],float);expected_axis=original_axis/np.linalg.norm(original_axis)
        _require(np.allclose(axis,expected_axis,rtol=0,atol=1e-12)and np.allclose(offset,np.asarray(raw['offset'])-center,rtol=0,atol=1e-12),
            'continuous axis/offset not in actual centered metric frame')
    return dict(actual_center_m=center.tolist(),ideal_C_m=C.tolist(),actual_C_checked=actual_C is not None,
        discrete_count=len(spec.discrete),continuous_count=len(spec.continuous),C_conjugation_passed=True,axis_offset_passed=True)


def prepare_frame(*,rgb,depth,expected_shape_hw,click_xy,K,depth_scale_m,mesh_mm,model_info_mm,output,source_units):
    """One independent sequence; RGB/depth bytes remain unchanged, no mask input."""
    output=Path(output);_require(not output.exists(),'fresh adapter output required')
    rgb_path=verify_source(rgb);depth_path=verify_source(depth);verify_source(mesh_mm)
    image=cv2.imread(str(rgb_path),cv2.IMREAD_COLOR);raw=cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED)
    shape=tuple(expected_shape_hw);point=np.asarray(click_xy)
    _require(image is not None and image.dtype==np.uint8 and image.shape==(*shape,3),'RGB shape/type differs')
    _require(raw is not None and raw.dtype==np.uint16 and raw.shape==shape,'registered raw uint16 depth shape/type differs')
    _require(point.shape==(2,)and point.dtype.kind in 'iu'and 0<=point[0]<shape[1]and 0<=point[1]<shape[0],'fixed integer current-image click required')
    K=np.asarray(K,dtype=np.float64);_require(K.shape==(3,3)and np.isfinite(K).all(),'per-image K required')
    _require(np.isfinite(depth_scale_m)and depth_scale_m>0,'explicit raw depth-to-metre scale required')
    metric_info=original_model_info_metres(model_info_mm)
    output.mkdir(parents=True);sequence=output/'sequence'
    for channel in ('rgb','depth'):(sequence/channel).mkdir(parents=True)
    name=rgb_path.name;_require(name==depth_path.name and name.endswith('.png'),'same original RGB/depth basename required')
    for channel,source,binding,decoded,mode in [('rgb',rgb_path,rgb,image,cv2.IMREAD_COLOR),
            ('depth',depth_path,depth,raw,cv2.IMREAD_UNCHANGED)]:
        target=sequence/channel/name
        with target.open('xb')as f:f.write(source.read_bytes())
        _require(sha256(target)==binding['sha256'],'copied input differs from original frozen binding')
        sealed=cv2.imdecode(np.frombuffer(target.read_bytes(),dtype=np.uint8),mode)
        _require(sealed is not None and identity(sealed)==identity(decoded),'decoded observation differs from pinned copied input')
    np.savetxt(sequence/'cam_K.txt',K,fmt='%.17g');_require(np.array_equal(np.loadtxt(sequence/'cam_K.txt'),K),'camera serialization changed')
    model=convert_original_cad(mesh_mm,output/'model_m.ply',source_units=source_units)
    _write_json(output/'model_info_m.json',metric_info)
    loader=product_loader_relation(output/'model_m.ply');symmetry=check_centered_symmetry(metric_info,loader['model_center_m'])
    receipt=dict(schema='bop_single_click_metric_input/1',status='prepared',rgb=rgb,depth=depth,original_name=name,
        shape_hw=list(shape),click_xy=point.tolist(),cam_K=K.tolist(),depth_scale_m_per_raw_unit=float(depth_scale_m),
        raw_depth=identity(raw),metric_depth=identity(raw.astype(np.float32)*depth_scale_m),
        input_RGB_depth_bytes_unchanged=True,model_conversion=model,product_default_loader=loader,
        metric_original_model_info=metric_info,symmetry_conversion=symmetry,GT_read=False,mask_input=False,GPU_run=False,
        static_single_frame=True,tracking_claimed=False)
    verify_source(rgb);verify_source(depth);verify_source(mesh_mm)  # Seal all original input bindings.
    _write_json(output/'receipt.json',receipt)
    return receipt
