"""
Bamboo-Mixer 启动脚本 (Paper Branch)
一键启动 API 服务和前端页面

使用方法:
    python start_server.py              # 启动完整服务
    python start_server.py --api-only    # 仅启动 API
    python start_server.py --download    # 仅下载模型
"""

import os
import sys
import argparse
import subprocess
import webbrowser
import time
import urllib.request

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def download_models():
    """下载模型文件"""
    print("=" * 50)
    print("开始下载模型...")
    print("=" * 50)

    # HuggingFace 仓库
    REPO_ID = "ByteDance-Seed/bamboo_mixer"
    BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"

    MODEL_FILES = [
        ("ckpts/mono/optimal.pt", "Mono 模型权重"),
        ("ckpts/formula/optimal.pt", "MolMix 模型权重"),
        ("ckpts/generator/diffusion.pt", "Diffusion 模型权重"),
        ("ckpts/generator/decoder.pt", "Decoder 模型权重"),
    ]

    models_dir = os.path.join(PROJECT_ROOT, "models", "bamboo_mixer", "ckpts")
    os.makedirs(models_dir, exist_ok=True)

    # 创建子目录
    os.makedirs(os.path.join(models_dir, "mono"), exist_ok=True)
    os.makedirs(os.path.join(models_dir, "formula"), exist_ok=True)
    os.makedirs(os.path.join(models_dir, "generator"), exist_ok=True)

    success_count = 0
    for file_path, description in MODEL_FILES:
        url = f"{BASE_URL}/{file_path}"
        filename = os.path.basename(file_path)
        dest_path = os.path.join(models_dir, file_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        print(f"\n下载 {description}: {filename}")
        print(f"目标: {dest_path}")

        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
            print(f"  已存在，跳过")
            success_count += 1
            continue

        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            urllib.request.urlretrieve(url, dest_path)
            size = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"  完成: {size:.2f} MB")
            success_count += 1
        except Exception as e:
            print(f"  失败: {e}")

    print("\n" + "=" * 50)
    print(f"下载完成: {success_count}/{len(MODEL_FILES)} 个文件")
    print("=" * 50)

    if success_count == len(MODEL_FILES):
        print("\n模型文件已就绪!")
        return True
    else:
        print("\n部分文件下载失败，请检查网络后重试")
        return False


def check_dependencies():
    """检查依赖是否安装"""
    print("检查依赖...")

    required = ['fastapi', 'uvicorn', 'torch', 'rdkit']
    missing = []

    for lib in required:
        try:
            __import__(lib)
            print(f"  [OK] {lib}")
        except ImportError:
            print(f"  [MISS] {lib} (未安装)")
            missing.append(lib)

    if missing:
        print(f"\n请安装缺失的依赖:")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True


def start_api_server():
    """启动 API 服务"""
    api_script = os.path.join(PROJECT_ROOT, "api_service.py")

    print("\n启动 API 服务...")
    print(f"API 文档: http://localhost:8003/docs")
    print("按 Ctrl+C 停止服务\n")

    subprocess.run([sys.executable, api_script], cwd=PROJECT_ROOT)


def start_frontend():
    """启动前端页面"""
    index_path = os.path.join(PROJECT_ROOT, "index.html")

    # 使用系统默认浏览器打开
    print("\n打开前端页面...")
    webbrowser.open(f"file://{index_path}")


def main():
    parser = argparse.ArgumentParser(description="Bamboo-Mixer 启动脚本 (Paper Branch)")
    parser.add_argument('--download', action='store_true', help='仅下载模型')
    parser.add_argument('--api-only', action='store_true', help='仅启动 API，不打开前端')
    parser.add_argument('--check', action='store_true', help='检查依赖')

    args = parser.parse_args()

    print("=" * 50)
    print("  Bamboo-Mixer 电解液配方平台 (Paper Branch)")
    print("=" * 50)

    if args.check:
        check_dependencies()
        return

    if args.download:
        download_models()
        return

    # 检查依赖
    if not check_dependencies():
        return

    # 检查模型是否存在
    mono_ckpt = os.path.join(PROJECT_ROOT, "models", "bamboo_mixer", "ckpts", "mono", "optimal.pt")
    if not os.path.exists(mono_ckpt):
        print("\n模型文件未找到!")
        print("请先运行以下命令下载模型:")
        print("  python start_server.py --download")
        response = input("\n是否现在下载模型? (y/n): ")
        if response.lower() == 'y':
            if not download_models():
                return
        else:
            return

    # 启动 API 服务
    if args.api_only:
        start_api_server()
    else:
        # 同时启动 API 和前端
        import threading

        # 在新线程中启动 API
        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()

        # 等待服务启动
        print("\n等待服务启动...")
        time.sleep(3)

        # 打开前端
        start_frontend()

        # 保持主线程
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n服务已停止")


if __name__ == "__main__":
    main()
