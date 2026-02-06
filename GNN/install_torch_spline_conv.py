#!/usr/bin/env python3
"""
安装torch-spline-conv预编译版本的脚本
"""

import subprocess
import sys
import platform

def get_pytorch_info():
    """获取PyTorch版本信息"""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        return version, cuda_available, cuda_version
    except ImportError:
        print("PyTorch未安装")
        return None, False, None

def get_python_version():
    """获取Python版本"""
    return f"{sys.version_info.major}.{sys.version_info.minor}"

def get_platform_info():
    """获取平台信息"""
    return platform.system().lower(), platform.machine()

def install_torch_spline_conv():
    """安装torch-spline-conv预编译版本"""
    pytorch_version, cuda_available, cuda_version = get_pytorch_info()
    python_version = get_python_version()
    platform_system, platform_machine = get_platform_info()
    
    if not pytorch_version:
        print("请先安装PyTorch")
        return
    
    print(f"PyTorch版本: {pytorch_version}")
    print(f"CUDA可用: {cuda_available}")
    print(f"Python版本: {python_version}")
    print(f"平台: {platform_system} {platform_machine}")
    
    # 构建wheel包URL
    if platform_system == "windows":
        if cuda_available:
            # 有CUDA的Windows版本
            wheel_url = f"https://data.pyg.org/whl/torch-{pytorch_version}+cu{cuda_version}/torch_spline_conv-1.2.1+pt{pytorch_version.replace('.', '')}cu{cuda_version.replace('.', '')}-cp{python_version.replace('.', '')}-cp{python_version.replace('.', '')}-win_amd64.whl"
        else:
            # CPU版本的Windows
            wheel_url = f"https://data.pyg.org/whl/torch-{pytorch_version}+cpu/torch_spline_conv-1.2.1+pt{pytorch_version.replace('.', '')}cpu-cp{python_version.replace('.', '')}-cp{python_version.replace('.', '')}-win_amd64.whl"
    else:
        print("目前只支持Windows平台")
        return
    
    print(f"下载URL: {wheel_url}")
    
    try:
        # 安装预编译的wheel包
        subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_url])
        print("torch-spline-conv安装成功！")
    except subprocess.CalledProcessError as e:
        print(f"安装失败: {e}")
        print("尝试使用pip直接安装...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-spline-conv"])
            print("torch-spline-conv安装成功！")
        except subprocess.CalledProcessError as e2:
            print(f"直接安装也失败: {e2}")

if __name__ == "__main__":
    install_torch_spline_conv()

