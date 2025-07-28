import os
import sys
import json
import trimesh
import numpy as np
import re
import concurrent.futures
import xml.etree.ElementTree as ET


##############################################################################
#                           1) 箭头生成逻辑 (仅URDF)                        #
##############################################################################

def insert_arrow_object_and_material(obj_file_path, arrow_variant):
    """
    1) 打开刚由 trimesh 导出的箭头 OBJ 文件(其中可能出现各种 usemtl/o 行)
    2) 移除所有 usemtl/o 行
    3) 在文件开头插入:
       o arrow_{arrow_variant}
       usemtl arrow_{arrow_variant}
    => 保证箭头OBJ只使用 arrow_{variant} 这个材质，不写任何 base_geom，且写在OBJ最前面。
    """
    if not os.path.isfile(obj_file_path):
        print(f"[insert_arrow_object_and_material] file not found: {obj_file_path}")
        return

    with open(obj_file_path, "r", encoding="utf-8", errors="ignore") as fin:
        lines = fin.readlines()

    # 只保留 (v/vt/vn/f) 等几何行，去掉所有 "usemtl"、"o "、"g " ...
    filtered_geom_lines = []
    for line in lines:
        s = line.strip().lower()
        # 若行以 'usemtl'/'o'/'g' 开头，就跳过(不保留)
        if s.startswith("usemtl") or s.startswith("o ") or s.startswith("g "):
            continue
        filtered_geom_lines.append(line)

    # 在顶部插入箭头 object+材质声明
    new_lines = [
        f"o arrow_{arrow_variant}\n",
        f"usemtl arrow_{arrow_variant}\n"
    ]
    # 然后把过滤后的几何行拼接起来
    new_lines.extend(filtered_geom_lines)

    with open(obj_file_path, "w", encoding="utf-8") as fout:
        fout.writelines(new_lines)

    print(f"[insert_arrow_object_and_material] => {obj_file_path} with only arrow_{arrow_variant} material")


# --------------------- Slider 箭头生成逻辑 ---------------------

def load_mesh_from_obj_slider(obj_name, base_path):
    mp = os.path.join(base_path,"textured_objs",f"{obj_name}.obj")
    if not os.path.exists(mp):
        print("[slider] not found:", mp)
        return None
    try:
        return trimesh.load_mesh(mp)
    except Exception as e:
        print("[slider] load fail:", e)
        return None

def get_largest_mesh_slider(obj_list, base_path):
    mm=[]
    for onm in obj_list:
        m= load_mesh_from_obj_slider(onm, base_path)
        if m: mm.append(m)
    if not mm:
        return None
    if len(mm)==1:
        return mm[0]
    maxv=0
    best=None
    for x in mm:
        vol = np.prod(x.bounding_box.extents)
        if vol>maxv:
            maxv=vol
            best=x
    return best

def generate_flat_arrow_along_line_slider(start_point,direction,length,
                                          width=0.05,arrow_width_factor=1.5,
                                          arrow_length_factor=2.0, arrow_style=None):
    import numpy as np
    direction= direction/np.linalg.norm(direction)
    is_tapered = arrow_style in ("tapered_rectangular","tapered_blunt")
    is_blunt   = arrow_style in ("blunt","tapered_blunt")
    if arrow_style is None:
        alf= arrow_length_factor
    elif arrow_style=="blunt":
        alf= arrow_length_factor*0.5
    elif arrow_style=="tapered_rectangular":
        alf= arrow_length_factor
    elif arrow_style=="tapered_blunt":
        alf= arrow_length_factor*0.5

    arrow_part_length = width*alf
    body_length = length-arrow_part_length
    if body_length<0:
        body_length= length*0.8
    arrow_tip_width= width*arrow_width_factor
    if body_length< arrow_tip_width:
        body_length= arrow_tip_width*1.2
        length= body_length+arrow_part_length
    body_start= start_point
    body_end  = start_point + direction*body_length
    arrow_start= body_end
    arrow_tip= start_point+ direction*length

    if np.allclose(direction,[0,0,1]):
        side_vector= np.array([1,0,0])
    else:
        side_vector= np.cross(direction,[0,0,1])
        n= np.linalg.norm(side_vector)
        if n<1e-9:
            side_vector= np.array([1,0,0])
        else:
            side_vector/=n

    half_w= width/2
    if is_tapered:
        shw= half_w*0.5
        ehw= half_w*1.5
        v0= body_start+ side_vector*shw
        v1= body_start- side_vector*shw
        v2= body_end  - side_vector*ehw
        v3= body_end  + side_vector*ehw
        if arrow_style=="tapered_rectangular":
            arrow_base_hw= ehw*1.3
        else:
            arrow_base_hw= ehw*1.5
    else:
        v0= body_start+ side_vector*half_w
        v1= body_start- side_vector*half_w
        v2= body_end  - side_vector*half_w
        v3= body_end  + side_vector*half_w
        arrow_base_hw= (width*arrow_width_factor)/2
    if is_blunt:
        arrow_tip= arrow_start+ direction*(arrow_part_length*0.5)

    v4= arrow_start+ side_vector*arrow_base_hw
    v5= arrow_tip
    v6= arrow_start- side_vector*arrow_base_hw

    vertices= np.array([v0,v1,v2,v3,v4,v5,v6])
    faces=[
        [0,1,2],
        [0,2,3],
        [4,5,6]
    ]
    rf= [f[::-1] for f in faces]
    faces.extend(rf)

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

def generate_arrow_for_slider_slider(joint_data, node, base_path, arrow_style=None):
    import numpy as np
    axis= joint_data.get("axis",{})
    limit= joint_data.get("limit",{})
    if "direction" not in axis or "origin" not in axis:
        print("[slider] axis不完整 => None")
        return None
    if "a" not in limit or "b" not in limit:
        print("[slider] limit不完整 => None")
        return None
    axis_dir= np.array(axis["direction"],dtype=float)
    if np.linalg.norm(axis_dir)<1e-9:
        return None
    axis_dir/= np.linalg.norm(axis_dir)
    la= limit["a"]
    lb= limit["b"]
    trans= lb-la
    if abs(trans)<1e-6:
        print("[slider] 移动0 => None")
        return None

    handle_mesh=None
    if node.get("children"):
        for c in node["children"]:
            if c.get("name")=="handle" and "objs" in c:
                hm= get_largest_mesh_slider(c["objs"], base_path)
                if hm:
                    handle_mesh= hm
                    break
    if handle_mesh is None and node.get("name")=="handle" and "objs" in node:
        handle_mesh= get_largest_mesh_slider(node["objs"], base_path)

    if not handle_mesh:
        if "objs" in node:
            fallback= get_largest_mesh_slider(node["objs"], base_path)
            if fallback:
                handle_mesh= fallback
            else:
                found=False
                for ch in node.get("children",[]):
                    if "objs" in ch:
                        fb= get_largest_mesh_slider(ch["objs"], base_path)
                        if fb:
                            handle_mesh= fb
                            found=True
                            break
                if not found:
                    return None
        else:
            return None

    start_point= handle_mesh.centroid
    arrow_dir= axis_dir if trans>=0 else -axis_dir
    arrow_len= abs(trans)

    return generate_flat_arrow_along_line_slider(
        start_point= start_point,
        direction= arrow_dir,
        length= arrow_len,
        width=0.25,
        arrow_width_factor=1.5,
        arrow_length_factor=1.5,
        arrow_style= arrow_style
    )


# --------------------- Hinge 箭头生成逻辑 ---------------------

def load_mesh_from_obj_hinge(obj_name, base_path):
    mp= os.path.join(base_path,"textured_objs",f"{obj_name}.obj")
    if not os.path.exists(mp):
        print("[hinge] not found:", mp)
        return None
    try:
        return trimesh.load_mesh(mp)
    except Exception as e:
        print("[hinge] load fail:", e)
        return None

def load_and_get_part_meshes_hinge(part_objs, base_path):
    arr=[]
    for onm in part_objs:
        m= load_mesh_from_obj_hinge(onm, base_path)
        if m: arr.append(m)
    if not arr:
        return None
    return trimesh.util.concatenate(arr)

def compute_max_distance_point_hinge(mesh, axis_origin, axis_direction):
    import numpy as np
    vs= mesh.vertices
    axis_direction/= np.linalg.norm(axis_direction)
    vecs= vs- axis_origin
    projs= np.dot(vecs, axis_direction)
    proj_points= axis_origin + np.outer(projs, axis_direction)
    dist= np.linalg.norm(vs- proj_points,axis=1)
    idx= np.argmax(dist)
    return vs[idx], dist[idx]

def compute_max_distance_edge_midpoint_hinge(mesh, axis_origin, axis_direction):
    if mesh.edges_unique.shape[0]==0:
        return compute_max_distance_point_hinge(mesh, axis_origin, axis_direction)
    ed= mesh.edges_unique
    vs= mesh.vertices
    mids= (vs[ed[:,0]]+ vs[ed[:,1]])/2
    axis_direction/= np.linalg.norm(axis_direction)
    vecs= mids- axis_origin
    projs= np.dot(vecs, axis_direction)
    pp= axis_origin+ np.outer(projs, axis_direction)
    dist= np.linalg.norm(mids- pp, axis=1)
    idx= np.argmax(dist)
    mp= mids[idx]
    md= dist[idx]
    if md<1e-6:
        return compute_max_distance_point_hinge(mesh, axis_origin, axis_direction)
    return mp, md

def find_handle_node_hinge(node):
    if node.get("name")=="handle":
        return node
    for c in node.get("children",[]):
        fd= find_handle_node_hinge(c)
        if fd: return fd
    return None

def find_longest_handle_obj_hinge(obj_names, base_path):
    import numpy as np
    longest_len=-1
    best_mesh=None
    best_name=None
    for nm in obj_names:
        mm= load_mesh_from_obj_hinge(nm, base_path)
        if mm:
            bmin,bmax= mm.bounds
            diag= np.linalg.norm(bmax-bmin)
            if diag>longest_len:
                longest_len= diag
                best_mesh= mm
                best_name= nm
    return best_mesh,best_name

def generate_flat_arrow_along_arc_hinge(center,normal,radius,angle_start,angle_end,
                                        width=0.05,arrow_width_factor=1.5,
                                        arrow_length_factor=2.0,num_points=50,
                                        start_point=None,arrow_style=None):
    import numpy as np
    is_tapered= arrow_style in ("tapered_rectangular","tapered_blunt")
    is_blunt  = arrow_style in ("blunt","tapered_blunt")
    if arrow_style is None:
        alf= arrow_length_factor
    elif arrow_style=="blunt":
        alf= arrow_length_factor*0.5
    elif arrow_style=="tapered_rectangular":
        alf= arrow_length_factor
    elif arrow_style=="tapered_blunt":
        alf= arrow_length_factor*0.5

    angles= np.linspace(angle_start, angle_end, num_points)
    normal= normal/ np.linalg.norm(normal)
    if start_point is not None:
        rv= start_point- center
        if np.linalg.norm(rv)<1e-9:
            rv= np.array([1,0,0])
        else:
            rv/= np.linalg.norm(rv)
    else:
        if np.allclose(normal,[0,0,1]):
            rv= np.array([1,0,0])
        else:
            rv= np.cross(normal,[0,0,1])
            n_= np.linalg.norm(rv)
            if n_<1e-9:
                rv= np.array([1,0,0])
            else:
                rv/=n_

    arc_points=[]
    for ang in angles:
        mat= trimesh.transformations.rotation_matrix(ang, normal, point=center)
        p= center+ rv*radius
        r= trimesh.transformations.transform_points([p],mat)[0]
        arc_points.append(r)
    arc_points= np.array(arc_points)
    if len(arc_points)<2:
        print("[hinge] arc不足 => None")
        return None

    _, uq_idx= np.unique(arc_points, axis=0, return_index=True)
    arc_points= arc_points[np.sort(uq_idx)]
    if len(arc_points)<2:
        return None

    if is_tapered:
        shw= (width/2)*0.5
        ehw= (width/2)*1.5
        arrow_base_scale=1.3
        if arrow_style=="tapered_blunt":
            arrow_base_scale=1.5
    else:
        shw= width/2
        ehw= width/2
        arrow_base_scale=1.0

    left_side=[]
    right_side=[]
    for i in range(len(arc_points)):
        if i==0 and len(arc_points)>1:
            tangent= arc_points[i+1]-arc_points[i]
        elif i>0:
            tangent= arc_points[i]- arc_points[i-1]
        else:
            continue
        tl= np.linalg.norm(tangent)
        if tl<1e-9:
            continue
        tangent/= tl
        t_ratio= i/(len(arc_points)-1)
        cur_hw= shw*(1-t_ratio)+ ehw*t_ratio
        lp= arc_points[i]+ normal* cur_hw
        rp= arc_points[i]- normal* cur_hw
        left_side.append(lp)
        right_side.append(rp)
    if len(left_side)<2 or len(right_side)<2:
        print("[hinge] 两侧不足 => None")
        return None

    tangent_end= arc_points[-1]- arc_points[-2]
    t_end_len= np.linalg.norm(tangent_end)
    if t_end_len<1e-9:
        return None
    tangent_end/= t_end_len

    arrow_tip= arc_points[-1]+ tangent_end*(width*alf)
    if is_blunt:
        arrow_tip= arc_points[-1]+ tangent_end*(width*alf*0.5)
    arrow_width_size= (width*arrow_width_factor)/2
    if is_tapered:
        arrow_width_size= ehw* arrow_base_scale

    arrow_left= arc_points[-1]+ normal*arrow_width_size
    arrow_right=arc_points[-1]- normal*arrow_width_size

    vertices= np.array(left_side+ right_side + [arrow_left, arrow_tip, arrow_right])
    n_verts= len(vertices)
    idx_left= n_verts-3
    idx_tip=  n_verts-2
    idx_right=n_verts-1
    n_arc= len(left_side)
    idx_left_end= n_arc-1
    idx_right_end= n_arc*2-1

    faces=[]
    for i in range(n_arc-1):
        v0= i
        v1= i+1
        v2= n_arc+i+1
        v3= n_arc+i
        faces.append([v0,v1,v3])
        faces.append([v1,v2,v3])

    faces.append([idx_left, idx_tip, idx_right])
    faces.append([idx_left_end, idx_left, idx_right_end])
    faces.append([idx_right_end, idx_left, idx_right])

    rf= [f[::-1] for f in faces]
    faces.extend(rf)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def generate_arrow_for_hinge_hinge(joint_data, part_mesh, node, base_path, arrow_style=None):
    import numpy as np
    from math import radians

    # 1) 读取并检查关节数据
    axis = joint_data.get("axis", {})
    limit = joint_data.get("limit", {})
    if "origin" not in axis or "direction" not in axis:
        return None
    if "a" not in limit or "b" not in limit:
        return None

    axis_origin = np.array(axis["origin"], dtype=float)
    axis_dir = np.array(axis["direction"], dtype=float)
    if np.linalg.norm(axis_dir) < 1e-9:
        return None
    axis_dir /= np.linalg.norm(axis_dir)  # 归一化

    # 2) 得到角度区间(弧度)
    a_s = radians(limit["a"])
    a_e = radians(limit["b"])

    # 3) 找一个箭头的起点（如果有把手，就用把手中心；否则退化到最大边缘中点）
    handle_node = find_handle_node_hinge(node)
    start_point = None
    if handle_node and "objs" in handle_node:
        meshH, nameH = find_longest_handle_obj_hinge(handle_node["objs"], base_path)
        if meshH is not None:
            start_point = meshH.centroid
            print("[hinge] handle=", nameH)
        else:
            print("[hinge] no handle => fallback to edge midpoint")
    else:
        print("[hinge] no handle node => fallback to edge midpoint")

    if start_point is None:
        mp, dist_ = compute_max_distance_edge_midpoint_hinge(part_mesh, axis_origin, axis_dir)
        start_point = mp

    # 4) 限制画弧最大角度
    import math
    max_deg = 40
    max_rad = math.radians(max_deg)
    diff = a_e - a_s
    if abs(diff) > max_rad:
        if diff > 0:
            a_e = a_s + max_rad
        else:
            a_e = a_s - max_rad

    # 5) 生成箭头网格
    radius = np.linalg.norm(start_point - axis_origin)
    arrow_mesh = generate_flat_arrow_along_arc_hinge(
        center=axis_origin,
        normal=axis_dir,
        radius=radius,
        angle_start=a_s,
        angle_end=a_e,
        width=0.25,
        arrow_width_factor=1.5,
        arrow_length_factor=1.5,
        num_points=100,
        start_point=start_point,
        arrow_style=arrow_style
    )

    return arrow_mesh


##############################################################################
#            2) OBJ合并逻辑(把非arrow材质改为 base_geom) + MTL合并            #
##############################################################################

def parse_face_indices(face_str):
    parts= face_str.split('/')
    v= int(parts[0]) if parts[0] else None
    vt= int(parts[1]) if len(parts)>1 and parts[1] else None
    vn= int(parts[2]) if len(parts)>2 and parts[2] else None
    return v,vt,vn

def read_obj_file_as_white_or_arrow(obj_path):
    """
    读取 OBJ 时：
      - 如果 usemtl 包含 'arrow' => 保留为 {原样} (arrow_original/blunt等)
      - 否则 => 'base_geom'
    其余 (v,vt,vn,f) 正常解析
    """
    data = {
        "mtl_file": None,
        "vertices": [],
        "texcoords": [],
        "normals": [],
        "faces": []
    }
    curr_mat= None
    with open(obj_path,"r",encoding='utf-8',errors='ignore') as f:
        for line in f:
            l= line.strip()
            if not l or l.startswith('#'):
                continue
            if l.lower().startswith("mtllib "):
                # 可忽略
                continue
            if l.lower().startswith("usemtl "):
                m= l.split(None,1)[1].strip().lower()
                if "arrow" in m:
                    curr_mat= m  # e.g. arrow_original, arrow_blunt, ...
                else:
                    curr_mat= "base_geom"
                continue
            if l.startswith('v '):
                _, x,y,z= l.split()
                data["vertices"].append((float(x), float(y), float(z)))
            elif l.startswith('vt '):
                items= l.split()
                if len(items)==3:
                    _,u,v= items
                    data["texcoords"].append((float(u),float(v)))
                elif len(items)==4:
                    _,u,v,w= items
                    data["texcoords"].append((float(u),float(v),float(w)))
            elif l.startswith('vn '):
                _,nx,ny,nz= l.split()
                data["normals"].append((float(nx),float(ny),float(nz)))
            elif l.startswith('f '):
                fi= l[2:].split()
                face_trip=[]
                for f_ in fi:
                    vv,vt_,vn_ = parse_face_indices(f_)
                    face_trip.append((vv,vt_,vn_))
                data["faces"].append({
                    "material": curr_mat,
                    "face_data": face_trip
                })
    return data

def merge_obj_files_as_white_or_arrow(obj_file_list):
    merged_vertices=[]
    merged_texcoords=[]
    merged_normals=[]
    merged_faces=[]
    v_off=0; vt_off=0; vn_off=0

    for objp in obj_file_list:
        obj_data= read_obj_file_as_white_or_arrow(objp)

        for vx,vy,vz in obj_data["vertices"]:
            merged_vertices.append((vx,vy,vz))
        for tc in obj_data["texcoords"]:
            merged_texcoords.append(tc)
        for nx,ny,nz in obj_data["normals"]:
            merged_normals.append((nx,ny,nz))

        for face_item in obj_data["faces"]:
            mat= face_item["material"]
            new_f= []
            for (v_,vt_,vn_) in face_item["face_data"]:
                new_v= (v_off+ v_) if v_ else None
                new_vt= (vt_off+ vt_) if vt_ else None
                new_vn= (vn_off+ vn_) if vn_ else None
                new_f.append((new_v,new_vt,new_vn))
            merged_faces.append({
                "material": mat,
                "face_data": new_f
            })

        v_off+= len(obj_data["vertices"])
        vt_off+= len(obj_data["texcoords"])
        vn_off+= len(obj_data["normals"])

    return merged_vertices, merged_texcoords, merged_normals, merged_faces

def write_merged_obj_as_white_or_arrow(out_path, mv,mt,mn,mf):
    """
    将合并后的数据写入obj：
       - 如果 material 中包含 'arrow' => 直接 usemtl 同名 (arrow_{id}等)
       - 否则 => usemtl base_geom
    """
    with open(out_path,"w",encoding='utf-8') as fout:
        fout.write("# Merged(arrow/base_geom)\n\n")
        curr_mat= None

        # v
        for x,y,z in mv:
            fout.write(f"v {x} {y} {z}\n")
        # vt
        for t_ in mt:
            if len(t_)==2:
                fout.write(f"vt {t_[0]} {t_[1]}\n")
            else:
                fout.write(f"vt {t_[0]} {t_[1]} {t_[2]}\n")
        # vn
        for nx,ny,nz in mn:
            fout.write(f"vn {nx} {ny} {nz}\n")

        for face_item in mf:
            mat= face_item["material"]
            if mat!=curr_mat:
                curr_mat= mat
                if "arrow" in str(curr_mat).lower():
                    fout.write(f"usemtl {mat}\n")
                else:
                    fout.write("usemtl base_geom\n")
            face_str= []
            for (v_,vt_,vn_) in face_item["face_data"]:
                if not v_:
                    continue
                if vt_ and vn_:
                    face_str.append(f"{v_}/{vt_}/{vn_}")
                elif vt_ and not vn_:
                    face_str.append(f"{v_}/{vt_}")
                elif not vt_ and vn_:
                    face_str.append(f"{v_}//{vn_}")
                else:
                    face_str.append(str(v_))
            if len(face_str)>=3:
                fout.write("f "+ " ".join(face_str)+"\n")
    print(f"[write_merged_obj_as_white_or_arrow] => {out_path}")


def merge_two_obj_files_as_white_or_arrow(objA, objB, out_obj):
    """
    把 objA 和 objB 合并写成 out_obj
      - 包含 'arrow' 的 usemtl 就保持arrow
      - 否则 => base_geom
    """
    if not os.path.exists(objA):
        print(f"[merge_two_obj_files_as_white_or_arrow] {objA} not exist => skip")
        return
    if not os.path.exists(objB):
        print(f"[merge_two_obj_files_as_white_or_arrow] {objB} not exist => skip")
        return
    mv,mt,mn,mf= merge_obj_files_as_white_or_arrow([objA,objB])
    write_merged_obj_as_white_or_arrow(out_obj, mv, mt, mn, mf)


##############################################################################
#          3) URDF解析和箭头生成 (无JSON依赖)                               #
##############################################################################

def parse_urdf_file(urdf_path):
    """
    解析URDF文件，提取关节和链接信息
    返回一个包含所有链接和关节信息的字典
    """
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        urdf_data = {
            "links": {},
            "joints": {}
        }
        
        # 解析链接
        for link in root.findall(".//link"):
            link_name = link.get("name")
            visuals = []
            
            for visual in link.findall(".//visual"):
                visual_name = visual.get("name", "")
                origin = visual.find("origin")
                geometry = visual.find(".//mesh")
                
                if geometry is not None and "filename" in geometry.attrib:
                    mesh_file = geometry.get("filename")
                    
                    # 提取原点信息
                    xyz = [0, 0, 0]
                    if origin is not None and "xyz" in origin.attrib:
                        xyz = [float(x) for x in origin.get("xyz").split()]
                    
                    visuals.append({
                        "name": visual_name,
                        "mesh_file": mesh_file,
                        "origin": xyz
                    })
            
            urdf_data["links"][link_name] = {
                "name": link_name,
                "visuals": visuals
            }
        
        # 解析关节
        for joint in root.findall(".//joint"):
            joint_name = joint.get("name")
            joint_type = joint.get("type")
            
            parent = joint.find("parent")
            child = joint.find("child")
            origin = joint.find("origin")
            axis = joint.find("axis")
            limit = joint.find("limit")
            
            parent_link = parent.get("link") if parent is not None else None
            child_link = child.get("link") if child is not None else None
            
            # 提取原点信息
            xyz = [0, 0, 0]
            rpy = [0, 0, 0]
            if origin is not None:
                if "xyz" in origin.attrib:
                    xyz = [float(x) for x in origin.get("xyz").split()]
                if "rpy" in origin.attrib:
                    rpy = [float(r) for r in origin.get("rpy").split()]
            
            # 提取轴信息
            axis_xyz = [0, 0, 1]  # 默认Z轴
            if axis is not None and "xyz" in axis.attrib:
                axis_xyz = [float(x) for x in axis.get("xyz").split()]
            
            # 提取限制信息
            limit_info = {}
            if limit is not None:
                if "lower" in limit.attrib:
                    limit_info["lower"] = float(limit.get("lower"))
                if "upper" in limit.attrib:
                    limit_info["upper"] = float(limit.get("upper"))
                if "effort" in limit.attrib:
                    limit_info["effort"] = float(limit.get("effort"))
                if "velocity" in limit.attrib:
                    limit_info["velocity"] = float(limit.get("velocity"))
            
            urdf_data["joints"][joint_name] = {
                "name": joint_name,
                "type": joint_type,
                "parent": parent_link,
                "child": child_link,
                "origin": {
                    "xyz": xyz,
                    "rpy": rpy
                },
                "axis": axis_xyz,
                "limit": limit_info
            }
        
        return urdf_data
    
    except Exception as e:
        print(f"解析URDF文件出错: {e}")
        return None

def build_node_from_urdf(urdf_data, link_name, parent_name=None):
    """
    从URDF数据构建节点树
    """
    if link_name not in urdf_data["links"]:
        return None
    
    link_data = urdf_data["links"][link_name]
    
    # 创建节点
    node = {
        "name": link_name,
        "objs": [],
        "children": []
    }
    
    # 添加网格文件
    for visual in link_data["visuals"]:
        mesh_file = visual["mesh_file"]
        if mesh_file.startswith("textured_objs/"):
            obj_name = mesh_file.replace("textured_objs/", "").replace(".obj", "")
            if obj_name not in node["objs"]:
                node["objs"].append(obj_name)
    
    # 查找子链接
    for joint_name, joint_data in urdf_data["joints"].items():
        if joint_data["parent"] == link_name:
            child_link = joint_data["child"]
            child_node = build_node_from_urdf(urdf_data, child_link, link_name)
            if child_node:
                node["children"].append(child_node)
    
    return node

def create_moveable_part_visualization(base_dir, joint_id, joint_name, matching_node):
    """
    为指定关节ID创建可动部件可视化
    
    参数:
        base_dir: 基础目录
        joint_id: 关节ID
        joint_name: 关节名称
        matching_node: 包含关节的节点
    
    返回:
        生成的PLY文件路径
    """
    visualization_folder = os.path.join(base_dir, "yy_visualization")
    os.makedirs(visualization_folder, exist_ok=True)
    
    # 加载整体模型
    obj_files = []
    textured_objs_dir = os.path.join(base_dir, "textured_objs")
    if os.path.exists(textured_objs_dir):
        for obj_file in os.listdir(textured_objs_dir):
            if obj_file.endswith(".obj"):
                obj_files.append(os.path.join(textured_objs_dir, obj_file))
    
    if not obj_files:
        print(f"[create_moveable_part_visualization] 没有找到OBJ文件: {textured_objs_dir}")
        return None
    
    # 加载整体模型
    try:
        meshes = []
        for obj_path in obj_files:
            try:
                mesh = trimesh.load(obj_path)
                meshes.append(mesh)
            except Exception as e:
                print(f"[create_moveable_part_visualization] 加载 {obj_path} 出错: {e}")
                continue
        
        if not meshes:
            print(f"[create_moveable_part_visualization] 没有成功加载任何网格")
            return None
        
        whole_mesh = trimesh.util.concatenate(meshes)
        print(f"[create_moveable_part_visualization] 成功加载整体模型，包含 {len(whole_mesh.faces)} 个面片")
    except Exception as e:
        print(f"[create_moveable_part_visualization] 加载整体模型出错: {e}")
        return None
    
    # 初始化所有面片为不可动(0)
    face_labels = np.zeros(len(whole_mesh.faces), dtype=np.int64)
    
    # 收集与关节相关的可动部件的OBJ文件
    moveable_objs = []
    
    # 函数用于收集节点及其所有子节点的OBJ文件
    def collect_obj_files(node):
        obj_files = []
        if "objs" in node:
            obj_files.extend(node["objs"])
        for child in node.get("children", []):
            obj_files.extend(collect_obj_files(child))
        return obj_files
    
    # 收集关节节点及其子节点的所有OBJ文件
    moveable_objs = collect_obj_files(matching_node)
    
    # 如果找到可动部件，将它们的面片标记为1
    if moveable_objs:
        print(f"[create_moveable_part_visualization] 关节 {joint_id} 找到 {len(moveable_objs)} 个可动部件OBJ")
        
        # 对每个可动部件OBJ文件，将其面片标记为可动(1)
        for obj_name in moveable_objs:
            mesh_path = os.path.join(base_dir, "textured_objs", f"{obj_name}.obj")
            if os.path.exists(mesh_path):
                try:
                    part_mesh = trimesh.load_mesh(mesh_path)
                    
                    # 使用质心匹配将部件面片映射到whole_mesh
                    whole_centroids = np.mean(whole_mesh.vertices[whole_mesh.faces], axis=1)
                    part_centroids = np.mean(part_mesh.vertices[part_mesh.faces], axis=1)
                    
                    for part_idx, part_centroid in enumerate(part_centroids):
                        # 计算到whole中所有面的距离
                        distances = np.linalg.norm(whole_centroids - part_centroid, axis=1)
                        # 找到最近的面的索引
                        closest_face_idx = np.argmin(distances)
                        # 如果距离小于阈值，认为匹配成功并标记为可动(1)
                        min_distance = distances[closest_face_idx]
                        if min_distance < 0.05:  # 距离阈值，可根据数据特性调整
                            face_labels[closest_face_idx] = 1
                    
                    print(f"[create_moveable_part_visualization] 处理 {obj_name}.obj 完成")
                except Exception as e:
                    print(f"[create_moveable_part_visualization] 加载 {obj_name}.obj 出错: {e}")
    else:
        print(f"[create_moveable_part_visualization] 关节 {joint_id} 没有找到可动部件")
        return None
    
    # 创建面片颜色
    face_colors = np.zeros((len(face_labels), 4), dtype=np.uint8)
    face_colors[face_labels == 0] = [255, 0, 0, 255]  # 红色表示不可动部分
    face_colors[face_labels == 1] = [0, 255, 0, 255]  # 绿色表示可动部分
    
    # 创建带颜色的网格
    colored_mesh = trimesh.Trimesh(
        vertices=whole_mesh.vertices,
        faces=whole_mesh.faces,
        face_colors=face_colors
    )
    
    # 导出PLY文件
    output_path = os.path.join(visualization_folder, f"moveable_{joint_id}.ply")
    colored_mesh.export(output_path)
    
    # 同时保存标签为TXT文件，每行一个面片的标签(0或1)
    labels_path = os.path.join(visualization_folder, f"labels_{joint_id}.txt")
    with open(labels_path, 'w') as f:
        for label in face_labels:
            f.write(f"{label}\n")
    
    moveable_count = np.sum(face_labels == 1)
    fixed_count = np.sum(face_labels == 0)
    
    print(f"[create_moveable_part_visualization] 保存可视化结果到 {output_path}")
    print(f"[create_moveable_part_visualization] 保存标签到 {labels_path}")
    print(f"[create_moveable_part_visualization] 统计: {moveable_count} 个可动面片, {fixed_count} 个固定面片")
    
    return output_path

def process_one_object_from_urdf(base_dir, fail_logger):
    """
    仅从URDF文件生成箭头和可视化，无需JSON文件
    最终生成:
       yy_arrow/yy_arrow_{id}.obj 为每个关节ID生成单独的箭头
    """
    print(f"\n=== 处理对象目录(URDF): {base_dir} ===")
    
    mobility_urdf_path = os.path.join(base_dir, "mobility.urdf")
    arrow_folder = os.path.join(base_dir, "yy_arrow")
    os.makedirs(arrow_folder, exist_ok=True)
    
    # 检查URDF文件是否存在
    if not os.path.exists(mobility_urdf_path):
        rs = f"[process_one_object_from_urdf] no mobility.urdf: {mobility_urdf_path}"
        print(rs)
        fail_logger.append({"object_dir": base_dir, "reason": rs})
        return False
    
    # 解析URDF文件
    urdf_data = parse_urdf_file(mobility_urdf_path)
    if not urdf_data:
        rs = f"[process_one_object_from_urdf] 解析URDF失败: {mobility_urdf_path}"
        print(rs)
        fail_logger.append({"object_dir": base_dir, "reason": rs})
        return False
    
    # 构建URDF关节映射
    urdf_joints_by_name = {}
    urdf_joints_by_child = {}
    
    for joint_name, joint_data in urdf_data["joints"].items():
        urdf_joints_by_name[joint_name] = joint_data
        if "child" in joint_data and joint_data["child"]:
            child_link = joint_data["child"]
            if child_link not in urdf_joints_by_child:
                urdf_joints_by_child[child_link] = []
            urdf_joints_by_child[child_link].append(joint_data)
    
    # 提取URDF中的handle信息
    handle_info_by_link = {}
    for link_name, link_data in urdf_data["links"].items():
        handle_visuals = []
        for visual in link_data["visuals"]:
            visual_name = visual.get("name", "")
            if "handle" in visual_name.lower():
                handle_visuals.append(visual)
        
        if handle_visuals:
            handle_info_by_link[link_name] = {
                "visuals": handle_visuals,
                "obj_names": []
            }
            
            # 提取handle的OBJ名称
            for visual in handle_visuals:
                mesh_file = visual.get("mesh_file", "")
                if mesh_file.startswith("textured_objs/"):
                    obj_name = mesh_file.replace("textured_objs/", "").replace(".obj", "")
                    if obj_name not in handle_info_by_link[link_name]["obj_names"]:
                        handle_info_by_link[link_name]["obj_names"].append(obj_name)
    
    # 构建根链接
    root_links = []
    for link_name in urdf_data["links"]:
        is_root = True
        for joint_data in urdf_data["joints"].values():
            if joint_data["child"] == link_name:
                is_root = False
                break
        if is_root:
            root_node = build_node_from_urdf(urdf_data, link_name)
            if root_node:
                root_links.append(root_node)
    
    # 合并层次结构中的节点
    merged_data = root_links
    
    # 从URDF中收集所有非固定关节（活动关节）
    active_urdf_joints = []
    for joint_name, joint_data in urdf_data["joints"].items():
        if joint_data["type"] not in ["fixed"]:
            joint_type = "hinge" if joint_data["type"] == "revolute" else "slider" if joint_data["type"] == "prismatic" else "unknown"
            if joint_type != "unknown":  # 只处理hinge和slider类型
                active_urdf_joints.append({
                    "name": joint_name,
                    "data": joint_data,
                    "type": joint_type
                })
    
    print(f"[process_one_object_from_urdf] 发现 {len(active_urdf_joints)} 个活动关节")
    
    # 为每个活动URDF关节分配一个ID并生成箭头
    generated_arrows = {}
    
    for i, urdf_joint in enumerate(active_urdf_joints):
        joint_id = str(i)  # 从0开始编号
        urdf_joint_data = urdf_joint["data"] 
        urdf_joint_name = urdf_joint["name"]
        joint_type = urdf_joint["type"]
        
        # 获取子链接名称
        child_link_name = urdf_joint_data.get("child")
        if not child_link_name:
            print(f"[process_one_object_from_urdf] 关节 {urdf_joint_name} 没有子链接")
            continue
        
        # 在层次结构中查找匹配的节点
        matching_node = None
        for root_node in merged_data:
            def find_node_by_name(node, target_name):
                if node.get("name") == target_name:
                    return node
                for child in node.get("children", []):
                    found = find_node_by_name(child, target_name)
                    if found:
                        return found
                return None
            
            matching_node = find_node_by_name(root_node, child_link_name)
            if matching_node:
                break
        
        if not matching_node:
            print(f"[process_one_object_from_urdf] 无法在层次结构中找到节点 {child_link_name}")
            continue
        
        # 创建关节数据
        joint_data = {}
        
        # 添加轴信息
        axis_xyz = urdf_joint_data.get("axis", [0, 0, 1])
        origin_xyz = urdf_joint_data.get("origin", {}).get("xyz", [0, 0, 0])
        
        joint_data["axis"] = {
            "direction": axis_xyz,
            "origin": origin_xyz
        }
        
        # 添加限制信息
        if "limit" in urdf_joint_data and urdf_joint_data["limit"]:
            joint_data["limit"] = {}
            
            if "lower" in urdf_joint_data["limit"]:
                joint_data["limit"]["a"] = urdf_joint_data["limit"]["lower"]
                if joint_type == "hinge":
                    # 弧度转角度，为hinge关节
                    joint_data["limit"]["a"] *= 180 / 3.14159
            else:
                joint_data["limit"]["a"] = 0
            
            if "upper" in urdf_joint_data["limit"]:
                joint_data["limit"]["b"] = urdf_joint_data["limit"]["upper"]
                if joint_type == "hinge":
                    # 弧度转角度，为hinge关节
                    joint_data["limit"]["b"] *= 180 / 3.14159
            else:
                joint_data["limit"]["b"] = 0 if joint_type == "slider" else 90  # 默认值
        else:
            # 如果没有limit信息，设置默认值
            joint_data["limit"] = {
                "a": 0,
                "b": 1.0 if joint_type == "slider" else 90  # slider默认1米，hinge默认90度
            }
        
        # 查找handle信息
        handle_node = None
        handle_objs = []
        
        # 1. 首先检查当前链接是否有handle信息
        if child_link_name in handle_info_by_link:
            handle_objs = handle_info_by_link[child_link_name]["obj_names"]
            print(f"[process_one_object_from_urdf] 在链接 {child_link_name} 中找到handle: {handle_objs}")
        
        # 2. 如果当前链接没有handle，查找子链接中的handle
        if not handle_objs:
            # 查找URDF中与当前关节相关的所有子链接
            for j_name, j_info in urdf_data["joints"].items():
                if j_info["parent"] == child_link_name:
                    child_of_child = j_info["child"]
                    if child_of_child in handle_info_by_link:
                        handle_objs = handle_info_by_link[child_of_child]["obj_names"]
                        print(f"[process_one_object_from_urdf] 在子链接 {child_of_child} 中找到handle: {handle_objs}")
                        break
        
        # 3. 如果在URDF中找不到handle，尝试在节点树中查找
        if not handle_objs:
            def find_handle_node_in_tree(node):
                if node.get("name", "").lower() == "handle":
                    return node
                for child in node.get("children", []):
                    handle = find_handle_node_in_tree(child)
                    if handle:
                        return handle
                return None
            
            handle_node = find_handle_node_in_tree(matching_node)
            if handle_node and "objs" in handle_node:
                handle_objs = handle_node.get("objs", [])
                print(f"[process_one_object_from_urdf] 在节点树中找到handle: {handle_objs}")
        
        # 如果找到了handle，将其添加到节点中
        if handle_objs:
            if "handle_objs" not in matching_node:
                matching_node["handle_objs"] = handle_objs
        
        # 生成箭头
        arrow = None
        if joint_type == "hinge":
            part_objs = matching_node.get("objs", [])
            
            # 如果找到了handle，创建一个包含handle信息的节点
            if handle_objs:
                handle_node = {"name": "handle", "objs": handle_objs}
                if "children" not in matching_node:
                    matching_node["children"] = []
                matching_node["children"].append(handle_node)
                print(f"[process_one_object_from_urdf] 为关节 {joint_id} 添加handle节点")
            
            pm = load_and_get_part_meshes_hinge(part_objs, base_dir)
            if pm:
                arrow = generate_arrow_for_hinge_hinge(joint_data, pm, matching_node, base_dir, arrow_style=None)
        elif joint_type == "slider":
            # 对于slider，也可以考虑使用handle信息
            if handle_objs:
                handle_node = {"name": "handle", "objs": handle_objs}
                if "children" not in matching_node:
                    matching_node["children"] = []
                matching_node["children"].append(handle_node)
                print(f"[process_one_object_from_urdf] 为关节 {joint_id} 添加handle节点")
            
            arrow = generate_arrow_for_slider_slider(joint_data, matching_node, base_dir, arrow_style=None)
            
        if arrow:
            generated_arrows[joint_id] = arrow
            print(f"[process_one_object_from_urdf] 成功为关节 {urdf_joint_name} (ID: {joint_id}, 类型: {joint_type}) 生成箭头")
            
            # 立即为该关节生成可视化
            try:
                create_moveable_part_visualization(
                    base_dir=base_dir,
                    joint_id=joint_id,
                    joint_name=urdf_joint_name,
                    matching_node=matching_node
                )
            except Exception as e:
                print(f"[process_one_object_from_urdf] 创建关节 {joint_id} 的可视化出错: {e}")
                
        else:
            print(f"[process_one_object_from_urdf] 无法为关节 {urdf_joint_name} (ID: {joint_id}, 类型: {joint_type}) 生成箭头")
    
    # 导出每个箭头
    for joint_id, arrow_mesh in generated_arrows.items():
        arrow_obj_path = os.path.join(arrow_folder, f"yy_arrow_{joint_id}.obj")
        try:
            arrow_mesh.export(arrow_obj_path)
            insert_arrow_object_and_material(arrow_obj_path, f"{joint_id}")
            print(f"[process_one_object_from_urdf] 导出箭头 => {arrow_obj_path}")
        except Exception as e:
            print(f"[process_one_object_from_urdf] 导出箭头 {joint_id} 失败: {e}")
    
    return bool(generated_arrows)  # 如果成功生成至少一个箭头，则返回True


##############################################################################
#          4) 最终生成：yy_object => yy_merged.obj + 带箭头.obj            #
##############################################################################

def final_merge_for_object(base_dir, fail_logger):
    """
    对单个对象：
      1) 从URDF生成箭头 => yy_arrow/yy_arrow_{id}.obj (只含箭头)
      2) 将 textured_objs/*.obj 合并 => yy_merged.obj (非arrow => base_geom)
      3) 再与每个箭头OBJ合并 => yy_arrow/yy_merged_{id}.obj
         (arrow部分依旧是 arrow_{id}，object部分是 base_geom，不会覆盖箭头)
    """
    # 1) 从URDF生成箭头和可视化
    process_one_object_from_urdf(base_dir, fail_logger)

    # 2) 合并 textured_objs => yy_merged.obj
    txtdir= os.path.join(base_dir,"textured_objs")
    if not os.path.isdir(txtdir):
        rs= f"[final_merge] no textured_objs => skip {base_dir}"
        print(rs)
        fail_logger.append({"object_dir":base_dir,"reason":rs})
        return

    obj_files= [os.path.join(txtdir,f) for f in os.listdir(txtdir) if f.lower().endswith(".obj")]
    if not obj_files:
        rs= f"[final_merge] no .obj in {txtdir}"
        print(rs)
        fail_logger.append({"object_dir":base_dir,"reason":rs})
        return

    merged_obj_path= os.path.join(base_dir,"yy_merged.obj")
    mv, mt, mn, mf= merge_obj_files_as_white_or_arrow(obj_files)
    write_merged_obj_as_white_or_arrow(merged_obj_path, mv, mt, mn, mf)

    # 3) 与每个箭头合并 => yy_arrow/yy_merged_{id}.obj
    arrow_dir= os.path.join(base_dir,"yy_arrow")
    if not os.path.isdir(arrow_dir):
        print(f"[final_merge] 箭头目录不存在: {arrow_dir}")
        return
        
    # 查找所有箭头文件
    arrow_files = [f for f in os.listdir(arrow_dir) if f.startswith("yy_arrow_") and f.endswith(".obj")]
    
    if not arrow_files:
        print(f"[final_merge] 未找到任何箭头文件在 {arrow_dir}")
        return
        
    # 为每个箭头生成合并文件，放到yy_arrow文件夹中
    for arrow_file in arrow_files:
        # 从文件名提取关节ID
        joint_id = arrow_file.replace("yy_arrow_", "").replace(".obj", "")
        arrow_path = os.path.join(arrow_dir, arrow_file)
        
        if os.path.exists(arrow_path):
            out_obj = os.path.join(arrow_dir, f"yy_merged_{joint_id}.obj")  # 放到yy_arrow文件夹中
            merge_two_obj_files_as_white_or_arrow(merged_obj_path, arrow_path, out_obj)
            print(f"[final_merge] => {out_obj}")
        else:
            print(f"[final_merge] 箭头文件不存在: {arrow_path}")
    
    print(f"[final_merge] 完成处理箭头与几何合并")
    
    print(f"[final_merge] 完成处理 {base_dir}")


##############################################################################
#                           5) 主函数 => 单对象 / 批量                         #
##############################################################################

def main():
    script_dir= os.path.dirname(os.path.abspath(__file__))
    partnet_root= os.path.abspath(os.path.join(script_dir,".."))
    urdf_dataset= os.path.join(partnet_root,"urdf")

    args= sys.argv[1:]
    object_dir_arg= None
    for i, ag in enumerate(args):
        if ag=="--object-dir" and (i+1< len(args)):
            object_dir_arg= args[i+1]
            break

    fail_logger=[]
    success_logger=[]

    def do_one_dir(obj_dir):
        local_fail=[]
        try:
            # 读取meta.json获取类别信息
            meta_path = os.path.join(obj_dir, "meta.json")
            category = "Unknown"
            model_id = os.path.basename(obj_dir)
            
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                        if "model_cat" in meta_data:
                            category = meta_data["model_cat"]
                except Exception as e:
                    print(f"[do_one_dir] 读取meta.json错误: {e}")
            
            # 处理对象
            final_merge_for_object(obj_dir, local_fail)
            
            # 如果没有失败记录，则添加到成功记录
            if not local_fail:
                return ([], [{
                    "object_dir": obj_dir,
                    "category": category,
                    "model_id": model_id,
                    "status": "success"
                }])
            else:
                # 添加类别信息到失败记录
                for item in local_fail:
                    item["category"] = category
                    item["model_id"] = model_id
                return (local_fail, [])
        except Exception as e:
            return ([{
                "object_dir": obj_dir,
                "reason": str(e),
                "category": category if 'category' in locals() else "Unknown",
                "model_id": model_id if 'model_id' in locals() else os.path.basename(obj_dir),
                "status": "failed"
            }], [])

    if object_dir_arg:
        obj_dir= os.path.abspath(object_dir_arg)
        if not os.path.isdir(obj_dir):
            print(f"[main] invalid --object-dir: {obj_dir}")
            fail_logger.append({"object_dir":obj_dir,"reason":"not a dir", "category":"Unknown", "model_id":os.path.basename(obj_dir)})
        else:
            print(f"[main] single => {obj_dir}")
            fails, successes = do_one_dir(obj_dir)
            fail_logger.extend(fails)
            success_logger.extend(successes)
    else:
        if not os.path.isdir(urdf_dataset):
            rs= f"[main] no urdf dataset directory => {urdf_dataset}"
            print(rs)
            fail_logger.append({"object_dir":urdf_dataset,"reason":rs, "category":"Unknown", "model_id":"urdf_dataset"})
        else:
            print(f"[main] 开始扫描目录: {urdf_dataset}")
            all_dirs = []
            try:
                all_dirs = [d for d in os.listdir(urdf_dataset) if os.path.isdir(os.path.join(urdf_dataset, d))]
                # 确保转换所有目录名为整数进行排序，以便按数字顺序处理
                all_dirs = sorted(all_dirs, key=lambda x: int(x) if x.isdigit() else float('inf'))
                print(f"[main] 找到 {len(all_dirs)} 个目录，包括：{all_dirs[:5]}...等")
            except Exception as e:
                print(f"[main] 扫描目录错误: {e}")
                fail_logger.append({"object_dir":urdf_dataset,"reason":f"扫描目录错误: {e}", "category":"Unknown", "model_id":"urdf_dataset"})
            
            print(f"[main] parallel => {len(all_dirs)} dirs")
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                ft_to_dir={}
                for sd in all_dirs:
                    p= os.path.join(urdf_dataset, sd)
                    if not os.path.isdir(p):
                        continue
                    ft= executor.submit(do_one_dir,p)
                    ft_to_dir[ft]= p

                total= len(ft_to_dir)
                done_count=0
                for ft in concurrent.futures.as_completed(ft_to_dir):
                    dd= ft_to_dir[ft]
                    done_count+=1
                    try:
                        fails, successes = ft.result()
                        if fails:
                            fail_logger.extend(fails)
                        if successes:
                            success_logger.extend(successes)
                        print(f"[main] done {dd} ({done_count}/{total})")
                    except Exception as e:
                        print(f"[main] error {dd}: {e}")
                        fail_logger.append({
                            "object_dir":dd,
                            "reason":str(e),
                            "category":"Unknown",
                            "model_id":os.path.basename(dd),
                            "status":"failed"
                        })

    # 保存JSON失败日志
    if fail_logger:
        failf= os.path.join(script_dir,"arrow_creation_failures.json")
        try:
            with open(failf,"w",encoding='utf-8') as f:
                json.dump(fail_logger,f,indent=4, ensure_ascii=False)
            print(f"[main] fails => {failf}")
        except Exception as e:
            print(f"[main] 写入失败日志出错: {e}")
    
    # 保存TXT分类成功和失败日志
    # 1. 按类别组织数据
    category_results = {}
    for item in success_logger:
        category = item["category"]
        if category not in category_results:
            category_results[category] = {"success": [], "failed": []}
        category_results[category]["success"].append(item["model_id"])
    
    for item in fail_logger:
        category = item.get("category", "Unknown")
        if category not in category_results:
            category_results[category] = {"success": [], "failed": []}
        category_results[category]["failed"].append({
            "model_id": item.get("model_id", "unknown"),
            "reason": item.get("reason", "Unknown reason")
        })
    
    # 2. 写入TXT报告
    report_path = os.path.join(script_dir, "arrow_creation_report.txt")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Arrow Creation Report (URDF Only)\n")
            f.write("==================================\n\n")
            
            # 总体统计
            f.write(f"Total processed: {len(success_logger) + len(fail_logger)}\n")
            f.write(f"Successful: {len(success_logger)}\n")
            f.write(f"Failed: {len(fail_logger)}\n\n")
            
            # 输出处理的ID范围
            all_ids = [int(item["model_id"]) for item in success_logger + fail_logger if item.get("model_id", "").isdigit()]
            if all_ids:
                f.write(f"ID range: {min(all_ids)} to {max(all_ids)}\n\n")
            
            f.write("Results by Category\n")
            f.write("------------------\n\n")
            
            # 按类别详细信息
            for category, data in sorted(category_results.items()):
                f.write(f"Category: {category}\n")
                f.write(f"  - Successful: {len(data['success'])}\n")
                f.write(f"  - Failed: {len(data['failed'])}\n\n")
                
                # 成功列表
                if data["success"]:
                    f.write("  Successful IDs:\n")
                    for model_id in sorted(data["success"], key=lambda x: int(x) if x.isdigit() else float('inf')):
                        f.write(f"    - {model_id}\n")
                    f.write("\n")
                
                # 失败列表
                if data["failed"]:
                    f.write("  Failed IDs:\n")
                    for item in sorted(data["failed"], key=lambda x: int(x["model_id"]) if x["model_id"].isdigit() else float('inf')):
                        f.write(f"    - {item['model_id']}: {item['reason']}\n")
                    f.write("\n")
                
                f.write("\n")
        
        print(f"[main] Report written to => {report_path}")
    except Exception as e:
        print(f"[main] 生成报告出错: {e}")
    
    # 再生成一个按ID排序的列表，便于查看
    id_sorted_report = os.path.join(script_dir, "arrow_creation_by_id.txt")
    try:
        with open(id_sorted_report, "w", encoding="utf-8") as f:
            f.write("Arrow Creation Report (Sorted by ID, URDF Only)\n")
            f.write("===============================================\n\n")
            
            # 创建一个所有ID的字典
            all_results = {}
            for item in success_logger:
                model_id = item["model_id"]
                all_results[model_id] = {
                    "status": "success",
                    "category": item["category"],
                    "reason": None
                }
            
            for item in fail_logger:
                model_id = item.get("model_id", "unknown")
                all_results[model_id] = {
                    "status": "failed",
                    "category": item.get("category", "Unknown"),
                    "reason": item.get("reason", "Unknown reason")
                }
            
            # 按ID排序输出
            for model_id in sorted(all_results.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
                result = all_results[model_id]
                if result["status"] == "success":
                    f.write(f"ID: {model_id} - ✓ Success - Category: {result['category']}\n")
                else:
                    f.write(f"ID: {model_id} - ✗ Failed - Category: {result['category']} - Reason: {result['reason']}\n")
        
        print(f"[main] ID-sorted report written to => {id_sorted_report}")
    except Exception as e:
        print(f"[main] 生成ID排序报告出错: {e}")
    
    if not fail_logger and not success_logger:
        print("[main] No processing done.")
    elif not fail_logger:
        print("[main] all success, no fails.")
    else:
        print(f"[main] {len(success_logger)} successes, {len(fail_logger)} failures.")

    print("[main] Done.")


if __name__=="__main__":
    main()
