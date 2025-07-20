# 虚拟扫描器修改计划：添加面片ID记录功能

## 目标
修改虚拟扫描器代码，在生成点云的同时记录每个点所对应的原始面片(三角形)的索引ID，以便后续分析和处理。

## 修改步骤

### 1. 修改 `virtual_scanner.h`

在 `VirtualScanner` 类中添加一个新的成员变量用于存储面片索引：

```cpp
// 添加到成员变量部分
vector<int> face_indices_;  // 存储每个点对应的面片索引
```

### 2. 修改 `virtual_scanner.cpp` 中的 `scanning` 方法

在处理射线与面片相交的代码中，收集面片ID：

```cpp
// 在 MatrixXf buffer_pt(3, Num), buffer_n(3, Num) 之后添加
VectorXi buffer_face_id = VectorXi::Constant(Num, -1);  // 默认值-1表示没有对应的面片

// 在找到交点后保存面片ID
// 在约400-450行，代码已经计算了id：
// id = std::distance(start_iteration, intersection->second);
// 修改为保存这个ID：
buffer_face_id(i) = id;

// 在收集点和法线的同时也收集面片ID
face_indices_.clear();
face_indices_.reserve(Num);
for (int i = 0; i < Num; ++i) {
  if (buffer_flag(i) != 0) {
    // 已有的点和法线处理代码...
    // 添加面片ID
    face_indices_.push_back(buffer_face_id(i));
  }
}
```

### 3. 修改 `points.h`

更新 `PtsInfo` 类以支持面片索引：

```cpp
enum PropType { 
  kPoint = 1, 
  kNormal = 2, 
  kFeature = 4, 
  kLabel = 8,
  kFaceIndex = 16  // 添加面片索引类型
};
static const int kPTypeNum = 5;  // 更新为5
```

### 4. 修改 `points.cpp`

更新 `Points::set_points` 方法，增加面片索引参数：

```cpp
bool Points::set_points(
    const vector<float>& pts, 
    const vector<float>& normals,
    const vector<float>& features = vector<float>(),
    const vector<float>& labels = vector<float>(),
    const vector<int>& face_indices = vector<int>())
{
  // 已有代码...
  
  // 添加面片索引处理
  if (!face_indices.empty()) {
    int c = face_indices.size() / num;
    int r = face_indices.size() % num;
    // 确保每个点对应一个面片索引
    if (1 != c || 0 != r) return false;
    info.set_channel(PtsInfo::kFaceIndex, c);
  }
  
  // 已有代码...
  
  // 复制面片索引数据
  if (!face_indices.empty()) {
    // 需要添加一个新方法来处理整型数据
    copy_int_data(face_indices.begin(), face_indices.end(), mutable_ptr_int(PtsInfo::kFaceIndex));
  }
  
  return true;
}

// 添加新方法处理整型数据
int* Points::mutable_ptr_int(PtsInfo::PropType ptype) {
  return reinterpret_cast<int*>(mutable_ptr(ptype));
}

void Points::copy_int_data(vector<int>::const_iterator begin, vector<int>::const_iterator end, int* dest) {
  std::copy(begin, end, dest);
}
```

### 5. 修改 `VirtualScanner::save_binary` 和 `VirtualScanner::save_ply` 方法

更新这些方法以保存面片索引数据：

```cpp
bool VirtualScanner::save_binary(const string& filename) {
  // 更新为包含面片索引
  bool succ = point_cloud_.set_points(pts_, normals_, vector<float>(), vector<float>(), face_indices_);
  // 其余代码不变...
}

bool VirtualScanner::save_ply(const string& filename) {
  // 修改PLY输出格式，包含面片ID
  // 在header中添加face_id属性
  outfile << "ply" << endl
      << "format ascii 1.0" << endl
      << "element vertex " << n << endl
      << "property float x" << endl
      << "property float y" << endl
      << "property float z" << endl
      << "property float nx" << endl
      << "property float ny" << endl
      << "property float nz" << endl
      << "property int face_id" << endl  // 添加face_id属性
      << "element face 0" << endl
      << "property list uchar int vertex_indices" << endl
      << "end_header" << endl;
  
  // 修改输出格式，包含面片ID
  for (int i = 0; i < n; ++i) {
    sprintf(pstr + i * len,
        "%.6f %.6f %.6f %.6f %.6f %.6f %d\n",  // 添加%d格式化面片ID
        pts_[3 * i], pts_[3 * i + 1], pts_[3 * i + 2],
        normals_[3 * i], normals_[3 * i + 1], normals_[3 * i + 2],
        face_indices_[i]);  // 输出面片ID
  }
  // 其余代码不变...
}
```

## 编译和测试

完成以上修改后，需要重新编译虚拟扫描器：

```bash
cd virtual_scanner_modi
mkdir -p build
cd build
cmake ..
make
```

然后使用修改后的虚拟扫描器处理3D模型，生成的点云文件将包含面片ID信息。

## 验证

处理一个测试模型后，可以检查生成的PLY文件，确认每个点是否都包含了对应的面片ID。

## 后续工作

1. 如果需要，可以添加一个新的函数来查询指定点对应的面片信息
2. 可以添加可视化工具，根据面片ID为点云上色，以便直观地验证结果
