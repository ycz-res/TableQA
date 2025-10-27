#!/bin/bash

# TbaleQA项目安装脚本

echo "🚀 开始安装TbaleQA项目..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ 错误: 需要Python 3.8或更高版本，当前版本: $python_version"
    exit 1
fi

echo "✅ Python版本检查通过: $python_version"

# 创建虚拟环境
echo "📦 创建虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 升级pip
echo "⬆️ 升级pip..."
pip install --upgrade pip

# 安装依赖
echo "📚 安装项目依赖..."
pip install -r requirements.txt

# 检查CUDA支持（可选）
echo "🔍 检查CUDA支持..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

echo "✅ 安装完成！"
echo ""
echo "🎯 使用方法:"
echo "1. 激活虚拟环境: source venv/bin/activate"
echo "2. 运行基础示例: python trl_training_example.py"
echo "3. 运行高级示例: python advanced_trl_examples.py"
echo ""
echo "📖 更多信息请查看README.md"

