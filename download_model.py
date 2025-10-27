"""
下载Qwen3-8B模型脚本
使用ModelScope（魔搭社区）进行下载 - 国内访问更快
"""

import os
import subprocess
from pathlib import Path

def download_qwen3_8b_modelscope(model_id="Qwen/Qwen2.5-7B-Instruct", save_dir="./models/pretrained"):
    """
    使用ModelScope（魔搭社区）下载Qwen模型到本地
    国内访问速度更快！
    
    参数:
        model_id: ModelScope模型ID，例如 "Qwen/Qwen2.5-7B-Instruct"
        save_dir: 保存目录
    """
    print(f"开始从ModelScope下载模型: {model_id}")
    print(f"保存路径: {save_dir}")
    
    # 创建保存目录
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    save_dir_abs = os.path.abspath(save_dir)
    
    try:
        # 检查并安装modelscope
        print("\n[1/3] 检查ModelScope库...")
        try:
            import modelscope
            print("✓ ModelScope已安装")
        except ImportError:
            print("正在安装ModelScope...")
            subprocess.check_call(["pip3", "install", "modelscope", "-q"])
            print("✓ ModelScope安装完成")
        
        # 导入modelscope
        from modelscope import snapshot_download
        
        print(f"\n[2/3] 从ModelScope下载模型...")
        print("这可能需要较长时间（模型约15GB），请耐心等待...\n")
        
        # 下载模型
        model_dir = snapshot_download(
            model_id,
            cache_dir=save_dir_abs,
            revision='master'
        )
        
        print(f"\n[3/3] 验证下载...")
        print(f"\n{'='*60}")
        print(f"✓ 模型下载成功！")
        print(f"保存位置: {model_dir}")
        print(f"{'='*60}")
        
        # 列出下载的文件
        print("\n已下载的主要文件:")
        model_path = Path(model_dir)
        total_size = 0
        for file in sorted(model_path.rglob("*")):
            if file.is_file() and not file.name.startswith('.'):
                size_mb = file.stat().st_size / (1024 * 1024)
                total_size += size_mb
                if size_mb > 1:  # 只显示大于1MB的文件
                    print(f"  {file.name} ({size_mb:.2f} MB)")
        
        print(f"\n总大小: {total_size / 1024:.2f} GB")
        
        return model_dir
            
    except Exception as e:
        print(f"\n❌ 下载失败: {str(e)}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 确保有足够的磁盘空间（需要约16GB）")
        print("3. 如果ModelScope访问失败，可以尝试:")
        print("   - 访问 https://modelscope.cn/ 注册账号")
        print(f"   - 或使用命令: modelscope download --model {model_id}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="下载Qwen模型（使用ModelScope国内源）")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="模型ID（默认: Qwen/Qwen2.5-1.5B-Instruct）。可选: Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-0.5B-Instruct等"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models/pretrained",
        help="保存目录（默认: ./models/pretrained）"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Qwen模型下载工具（ModelScope国内源）")
    print("="*60)
    print("\n提示: ModelScope是阿里云提供的模型平台，国内访问速度快！")
    
    download_qwen3_8b_modelscope(args.model_id, args.save_dir)

