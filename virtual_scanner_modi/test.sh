#!/bin/bash
# 测试脚本：验证面片ID是否被正确记录

# 确保在virtual_scanner_modi目录
cd $(dirname $0)

# 测试目录和文件
TEST_DIR="test_data"
mkdir -p ${TEST_DIR}

# 确保build/virtualscanner存在
if [ ! -f "build/virtualscanner" ]; then
    echo "错误：找不到可执行文件 build/virtualscanner"
    echo "请先运行 ./build.sh 编译程序"
    exit 1
fi

# 查找测试用的obj或off文件
TEST_OBJ=$(find data/original_obj -name "*.obj" -type f | head -1)
if [ -z "$TEST_OBJ" ]; then
    # 如果没找到，创建一个简单的立方体OBJ
    TEST_OBJ="${TEST_DIR}/cube.obj"
    echo "找不到测试用的OBJ文件，创建一个简单的立方体：${TEST_OBJ}"
    
    cat > ${TEST_OBJ} << EOL
# 立方体
v -1 -1 -1
v -1 -1  1
v -1  1 -1
v -1  1  1
v  1 -1 -1
v  1 -1  1
v  1  1 -1
v  1  1  1
f 1 2 4
f 1 4 3
f 5 7 8
f 5 8 6
f 1 5 6
f 1 6 2
f 3 4 8
f 3 8 7
f 1 3 7
f 1 7 5
f 2 6 8
f 2 8 4
EOL
fi

echo "使用测试文件: ${TEST_OBJ}"

# 生成点云文件
OUTPUT_PLY="${TEST_DIR}/output.ply"
echo "开始生成点云文件..."
./build/virtualscanner "${TEST_OBJ}" 6 0 1

# 获取生成的PLY文件路径
GENERATED_PLY="${TEST_OBJ%.obj}.ply"
if [ ! -f "${GENERATED_PLY}" ]; then
    echo "错误：未能生成PLY文件"
    exit 1
fi

# 复制到测试目录
cp "${GENERATED_PLY}" "${OUTPUT_PLY}"
echo "PLY文件已生成：${OUTPUT_PLY}"

# 检查PLY文件中是否包含face_id属性
echo "检查PLY文件中的face_id属性..."
if grep -q "property int face_id" "${OUTPUT_PLY}"; then
    echo "✓ 文件包含face_id属性"
    
    # 提取前10行数据查看
    echo "前10个点的数据（包含面片ID）:"
    grep -v "^[a-z]" "${OUTPUT_PLY}" | head -10
    
    echo "测试成功！"
else
    echo "✗ 文件不包含face_id属性"
    echo "测试失败：未找到面片ID信息"
    exit 1
fi 