# 虚拟扫描器 - 改进版

这个版本的虚拟扫描器在原始功能基础上增加了记录面片ID的功能。当生成点云时，每个点都会记录它所对应的原始模型的面片(三角形)ID。

## 改进内容

1. 在点云中添加了面片ID信息
2. 在PLY输出格式中增加了face_id属性
3. 修改了二进制格式以存储面片ID信息

## 编译方法

```bash
./build.sh
```

或手动编译：

```bash
mkdir -p build
cd build
cmake ..
make -j4
```

## 使用方法

与原始虚拟扫描器相同：

```bash
./virtualscanner <file_name> [nviews] [flags] [normalize]
```

参数说明：
- file_name: 要处理的文件(*.obj; *.off)
- nviews: 扫描视图数量，默认为6
- flags: 是否输出法线翻转标志，默认为0
- normalize: 是否标准化输入网格，默认为0

## 测试

运行测试脚本验证面片ID是否被正确记录：

```bash
./test.sh
```

## 数据格式

改进后的PLY格式包含额外的face_id属性：

```
ply
format ascii 1.0
element vertex <num_points>
property float x
property float y
property float z
property float nx
property float ny
property float nz
property int face_id
element face 0
property list uchar int vertex_indices
end_header
<x> <y> <z> <nx> <ny> <nz> <face_id>
...
```

其中 face_id 是点所对应的原始模型中面片的索引。

## 使用场景

此功能可用于：

1. 点云与原始网格的对应分析
2. 纹理或颜色从网格到点云的转移
3. 点云分割和分类的标注
4. 反投影和重建算法的验证 