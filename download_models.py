"""
模型下载脚本 (Paper Branch)
使用方法: python download_models.py
"""
import os
import urllib.request

# HuggingFace 仓库信息
REPO_ID = "ByteDance-Seed/bamboo_mixer"
BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"

# 需要下载的文件列表 (HuggingFace路径 -> 本地路径)
MODEL_FILES = [
    # 模型权重 (HuggingFace -> 本地)
    ("ckpts/mono/optimal.pt", "models/bamboo_mixer/ckpts/mono/optimal.pt"),
    ("ckpts/formula/optimal.pt", "models/bamboo_mixer/ckpts/formula/optimal.pt"),
    ("ckpts/generator/diffusion.pt", "models/bamboo_mixer/ckpts/generator/diffusion.pt"),
    ("ckpts/generator/decoder.pt", "models/bamboo_mixer/ckpts/generator/decoder.pt"),
    ("ckpts/generator/predictor.pt", "models/bamboo_mixer/ckpts/generator/predictor.pt"),
]

def download_file(url, dest_path):
    """下载单个文件"""
    print(f"下载: {dest_path}")
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        urllib.request.urlretrieve(url, dest_path)
        size = os.path.getsize(dest_path) / 1024 / 1024
        print(f"  完成: {size:.2f} MB")
        return True
    except Exception as e:
        print(f"  失败: {e}")
        return False

def main():
    # 项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))

    print(f"开始下载模型...")
    print(f"项目目录: {project_root}")
    print("=" * 50)

    success_count = 0
    for hf_path, local_path in MODEL_FILES:
        url = f"{BASE_URL}/{hf_path}"
        dest = os.path.join(project_root, local_path)

        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"跳过 (已存在): {local_path}")
            success_count += 1
            continue

        if download_file(url, dest):
            success_count += 1

    print("=" * 50)
    print(f"下载完成: {success_count}/{len(MODEL_FILES)} 个文件")

    if success_count == len(MODEL_FILES):
        print("\n模型文件列表:")
        models_dir = os.path.join(project_root, "models", "bamboo_mixer")
        for root, dirs, files in os.walk(models_dir):
            for f in files:
                if f.endswith('.pt'):
                    path = os.path.join(root, f)
                    size = os.path.getsize(path) / 1024 / 1024
                    rel_path = os.path.relpath(path, models_dir)
                    print(f"  {rel_path}: {size:.2f} MB")

if __name__ == "__main__":
    main()
