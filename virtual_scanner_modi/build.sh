#!/bin/bash
# 编译虚拟扫描器

# 确保在virtual_scanner_modi目录
cd $(dirname $0)

# 清除之前的构建
echo "清除旧的构建文件..."
rm -rf build
echo "创建新的build目录..."
mkdir -p build
cd build

# 生成CMake构建文件
echo "配置CMake..."
cmake ..

# 编译
echo "开始编译..."
make -j4

# 检查编译结果
if [ -f "virtualscanner" ]; then
    echo "编译成功！可执行文件位于 $(pwd)/virtualscanner"
    echo "用法示例: ./virtualscanner ../data/original_obj/model.obj 14 0 1"
    echo "参数说明: <文件名> [视图数=6] [法线标志=0] [标准化=0]"
else
    echo "编译失败，请检查错误信息。"
fi

cd .. 