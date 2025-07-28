1. 数据zip解压到urdf文件夹。
然后用keep_want_data_id.py删除一些不想要的id文件，脚本会根据
model_ids.txt删除一些文件。

2. 接着我们运行cleanup_model_folders.py移除不想要的文件内容

3. 我们进入scripts，运行convert_urdf_joints.py

4. 运行yy_arrow_creation_and_merge_urdf.py，生成arrow，visulization和merge
(到这一步，我复制了一个urdf_copy)

5. 我接着把数据复制到data_small/urdf

6. 接着我开始运行virtual scanner下面的yy_keep_structure.py

7. 接着我们开始运行PartField_Sketch_simpleMLP/Partfield/yy_partfield_inference.py，去获得feature。
先路径用urdf，再改成urdf_shell，从而对两种都获得feature

7. 目前都是正确的运行，我们得到了：urdf_original， urdf_shell, urdf_visulaization