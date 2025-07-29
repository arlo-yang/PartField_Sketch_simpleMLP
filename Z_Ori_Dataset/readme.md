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

8. 运行7_shell_mesh_gt.py我希望能够得到shell mesh的gt但是失败了。这个脚本有问题 可能我的思路也有问题。面片的几何结构
变化了。似乎点也没有align。不管了时间来不及了，我们先处理original的训练

9. img_pred，npy应该不需要动 这俩是图片域的,之前通过/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/segmentation2/all_confidence.py生成，是固定的问题不大

10. 可以check一下是不是面片confidence和几何体对应上。不要搞乱了 

11. 运行samesh_modi 的projection得到result，我们获得面片的confidence

12. 我们现在的data_small有： imr_pred, npy, urdf, result

13. 运行data_small下面的mesh_point.py。获得每一个面片对应的3个顶点的3D坐标

14. 我们需要partfield + confidence + position --> 0 / 1 face prediction,
可以看到/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small_train/NOTE.md,这里写了具体的操作，
以及一个复制的脚本copy_training_data.py