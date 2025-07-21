#!/bin/bash
# 批处理脚本：处理所有OBJ文件并生成面片ID

# 确保在virtual_scanner_modi目录
cd $(dirname $0)

# 输入和输出目录
INPUT_DIR="data/original_obj"
OUTPUT_DIR="data/id"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 确保build/virtualscanner存在
if [ ! -f "build/virtualscanner" ]; then
    echo "错误：找不到可执行文件 build/virtualscanner"
    echo "请先运行 ./build.sh 编译程序"
    exit 1
fi

# 检查输入目录是否存在
if [ ! -d "${INPUT_DIR}" ]; then
    echo "错误：输入目录 ${INPUT_DIR} 不存在"
    exit 1
fi

# 查找所有OBJ文件
OBJ_FILES=$(find ${INPUT_DIR} -name "*.obj" -type f)
if [ -z "${OBJ_FILES}" ]; then
    echo "警告：在 ${INPUT_DIR} 中没有找到OBJ文件"
    exit 1
fi

echo "找到以下OBJ文件："
echo "${OBJ_FILES}"
echo "------------------------------"

# 统计文件数量
FILE_COUNT=$(echo "${OBJ_FILES}" | wc -l)
echo "共找到 ${FILE_COUNT} 个OBJ文件"
echo "开始处理..."

# 计数器
CURRENT=0

# 批处理所有文件
for OBJ_FILE in ${OBJ_FILES}; do
    CURRENT=$((CURRENT + 1))
    FILENAME=$(basename "${OBJ_FILE}")
    MODEL_NAME="${FILENAME%.*}"
    
    echo "[${CURRENT}/${FILE_COUNT}] 处理: ${FILENAME}"
    
    # 运行virtualscanner生成点云和面片ID
    ./build/virtualscanner "${OBJ_FILE}" 14 0 1
    
    # 检查是否生成了面片ID文件
    GENERATED_TXT="${OBJ_FILE%.obj}_face_ids.txt"
    if [ ! -f "${GENERATED_TXT}" ]; then
        echo "  警告：未能为 ${FILENAME} 生成面片ID文件"
        continue
    fi
    
    # 移动面片ID文件到输出目录
    OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_face_ids.txt"
    cp "${GENERATED_TXT}" "${OUTPUT_FILE}"
    echo "  ✓ 面片ID已保存到: ${OUTPUT_FILE}"
done

echo "------------------------------"
echo "处理完成！共处理了 ${CURRENT} 个文件"
echo "结果保存在: ${OUTPUT_DIR}" 