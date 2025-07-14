import os
import shutil
import glob
from pathlib import Path

def copy_model_data():
    """
    查找并复制模型数据:
    - 从: /hy-tmp/Sketch2Art_data_web/javascript/urdf/{id}/yy_object/yy_move/
    - 到: /hy-tmp/PartField/yy_finetune_data/{id}/
    """
    # 定义源路径和目标路径基准
    source_base = "/hy-tmp/Sketch2Art_data_web/javascript/urdf"
    target_base = "/hy-tmp/PartField/yy_finetune_data"
    
    # 确保目标基础路径存在
    os.makedirs(target_base, exist_ok=True)
    
    # 查找所有可用的模型ID (只查找包含yy_move目录的模型)
    model_ids = []
    for item in os.listdir(source_base):
        yy_move_path = os.path.join(source_base, item, "yy_object", "yy_move")
        if os.path.isdir(yy_move_path):
            model_ids.append(item)
    
    print(f"找到 {len(model_ids)} 个可用模型")
    
    # 复制每个模型的数据
    success_count = 0
    fail_count = 0
    
    for model_id in model_ids:
        try:
            source_dir = os.path.join(source_base, model_id, "yy_object", "yy_move")
            target_dir = os.path.join(target_base, model_id)
            
            # 确保目标目录存在
            os.makedirs(target_dir, exist_ok=True)
            
            # 查找所有需要复制的文件
            obj_files = glob.glob(os.path.join(source_dir, "*.obj"))
            
            # 检查是否找到文件
            if not obj_files:
                print(f"警告: 模型 {model_id} 没有OBJ文件")
                fail_count += 1
                continue
                
            # 复制文件
            base_found = False
            moveable_found = False
            whole_found = False
            
            for obj_file in obj_files:
                file_name = os.path.basename(obj_file)
                target_file = os.path.join(target_dir, file_name)
                
                # 检查文件类型
                if file_name == "base.obj":
                    base_found = True
                elif file_name.startswith("moveable"):
                    moveable_found = True
                elif file_name == "whole.obj":
                    whole_found = True
                
                # 复制文件
                shutil.copy2(obj_file, target_file)
            
            # 检查是否找到了必要的文件
            if not base_found:
                print(f"警告: 模型 {model_id} 没有base.obj文件")
            if not moveable_found:
                print(f"警告: 模型 {model_id} 没有moveable*.obj文件")
            if not whole_found:
                print(f"警告: 模型 {model_id} 没有whole.obj文件")
            
            print(f"成功复制模型 {model_id}: {len(obj_files)} 个文件 (base: {base_found}, moveable: {moveable_found}, whole: {whole_found})")
            success_count += 1
            
        except Exception as e:
            print(f"处理模型 {model_id} 时出错: {str(e)}")
            fail_count += 1
    
    print("\n复制完成:")
    print(f"成功: {success_count} 个模型")
    print(f"失败: {fail_count} 个模型")

if __name__ == "__main__":
    copy_model_data()