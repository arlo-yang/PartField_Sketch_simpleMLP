# Virtual scanner for converting 3D model to point cloud


This folder contains the code for converting the 3D models to dense point clouds with normals (\*.points). As detailed in our paper, we build a virtual scanner and shoot rays to calculate the intersection point and oriented normal. 

The code is based on [Boost](https://www.boost.org/), [CGAL](http://www.cgal.org/) and the [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) libraries. After configuring these three libraries properly, the code can be built with visual studio easily.

`Note`: 
1. Sometimes, the executive file might collapse when the scale of the mesh is very large. This is one bug of CGAL. In order to mitigate this you can run VirtualScanner with the normalize flag set to 1.
2. The format of some off files in the `ModelNet40` is invalid. Before running the virtual scanner, fix the `ModelNet40` with this [script](https://github.com/Microsoft/O-CNN/blob/master/ocnn/octree/python/ocnn/utils/off_utils.py).


## Running Virtual Scanner
### Executable
The pre-built executive file is contained in the folder `prebuilt_binaries`, which has been test on the Win10 x64 system.

    Usage:
        VirtualScanner.exe <file_name> [nviews] [flags] [normalize]
            file_name: the name of the file (*.obj; *.off) to be processed.
            nviews: the number of views for scanning. Default: 6
            flags: Indicate whether to output normal flipping flag. Default: 0
            normalize: Indicate whether to normalize input mesh. Default: 0
    Example:
        VirtualScanner.exe input.obj 14         // process the file input.obj
        VirtualScanner.exe D:\data\ 14          // process all the obj/off files under the folder D:\Data


### Python
You can also use python to convert 'off' and 'obj' files.

    Example usage (single file):
        # Converts obj/off file to points
        from ocnn.virtualscanner import VirtualScanner
        scanner = VirtualScanner(filepath="input.obj", view_num=14, flags=False, normalize=True)
        scanner.save(output_path="output.points")

    Example usage (directory tree):
        # Converts all obj/off files in directory tree to points
        from ocnn.virtualscanner import DirectoryTreeScanner
        scanner = DirectoryTreeScanner(view_num=6, flags=False, normalize=True)
        scanner.scan_tree(input_base_folder='/ModelNet10', output_base_folder='/ModelNet10Points', num_threads=8)


## Output of Virtual Scanner
The result is in the format of `points`, which can be parsed with the following:

### CPP
```cpp
#include "virtual_scanner/points.h"

// ...
// Specify the filename of the points
string filename = "your_pointcloud.points";

// Load points
Points points;
points.read_points(filename)

// Point number
int n =  points.info().pt_num();

// Whether does the file contain point coordinates?
bool has_points = points.info().has_property(PtsInfo::kPoint);
// Get the pointer to points: x_1, y_1, z_1, ..., x_n, y_n, z_n
const float* ptr_points = points.ptr(PtsInfo::kPoint);

// Whether does the file contain normals?
bool has_normals = points.info().has_property(PtsInfo::kNormal);
// Get the pointer to normals: nx_1, ny_1, nz_1, ..., nx_n, ny_n, nz_n
const float* ptr_points = points.ptr(PtsInfo::kNormal);

// Whether does the file contain per-point labels?
bool has_labels = points.info().has_property(PtsInfo::kLabel);
// Get the pointer to labels: label_1, label_2, ..., label_n
const float* ptr_labels = points.ptr(PtsInfo::kLabel);
```

### Python
The Microsoft repo [O-CNN](https://github.com/Microsoft/O-CNN) contains the `ocnn.ocnn_base` package which defines a `Points` class under `ocnn.octree`. You can use this class to manipulate the points files and generate octrees.

## Building/Installing
### Building On Windows
To build in Windows you can,

1. Edit the project files to point to Boost, Eigen and CGAL,

or 

2. Use [Vcpkg](https://github.com/Microsoft/vcpkg) to install/build all the dependencies (note this takes a long time).
  ```
  git clone https://github.com/Microsoft/vcpkg
  cd vcpkg
  .\bootstrap-vcpkg.bat
  .\vcpkg integrate install
  .\vcpkg install cgal eigen3 boost-system boost-filesystem --triplet x64-windows
  ```
  Then to build, you can use the supplied solution file VirtualScanner.sln


### Building On Ubuntu
To build with ubuntu, you can use apt for the dependencies.
```
apt-get install -y --no-install-recommends libboost-all-dev libcgal-dev libeigen3-dev
```
Then you can use cmake to build the executable
From this project's directory,
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
### Installing Python Package
To install the python package you need to ensure that Eigen and CGAL can be found by cmake. If you used Vcpkg or apt-get to retrieve those libraries it should automatically find it.

With that ensured,

**Dependencies install via VCPKG**
```
pip install scikit-build cmake Cython
pip install --install-option="--" --install-option="-DCMAKE_TOOLCHAIN_FILE=<VCPKG_DIRECTORY>\scripts\buildsystems\vcpkg.cmake" .
```
Where <VCPKG_DIRECTORY> is the directory you install VCPKG.

**Dependencies install via apt-get**
```
pip install scikit-build cmake Cython
pip install .
```

If you use our code, please cite our paper.

    @article {Wang-2017-OCNN,
        title     = {O-CNN: Octree-based Convolutional Neural Networks for 3D Shape Analysis},
        author    = {Wang, Peng-Shuai and Liu, Yang and Guo, Yu-Xiao and Sun, Chun-Yu and Tong, Xin},
        journal   = {ACM Transactions on Graphics (SIGGRAPH)},
        volume    = {36},
        number    = {4},
        year      = {2017},
    }

# 虚拟扫描器 - 改进版

这个版本的虚拟扫描器在原始功能基础上增加了记录面片ID的功能。当生成点云时，每个点都会记录它所对应的原始模型的面片(三角形)ID。

## 改进内容

1. 在点云中添加了面片ID信息
2. 在PLY输出格式中增加了face_id属性
3. 修改了二进制格式以存储面片ID信息
4. 新增了导出面片ID到TXT文本文件的功能

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

### 改进后的PLY格式
包含额外的face_id属性：

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

### 新增的面片ID文本文件
同时会生成一个独立的TXT文件，包含点云与面片的对应关系：

```
# 点云与面片ID对应关系
# 格式: 点索引, 面片ID, x, y, z
# -------------------------------
0, 8, -0.484297, -0.996231, 0.520669
1, 8, -0.484297, -0.988694, 0.520669
...
```

其中 face_id 是点所对应的原始模型中面片的索引。

## 使用场景

此功能可用于：

1. 点云与原始网格的对应分析
2. 纹理或颜色从网格到点云的转移
3. 点云分割和分类的标注
4. 反投影和重建算法的验证 