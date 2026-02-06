#!/usr/bin/env python3
"""
修复conda网络连接问题的脚本
"""

import subprocess
import sys
import os

def run_command(cmd):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def fix_conda_channels():
    """修复conda频道配置"""
    print("正在修复conda网络连接问题...")
    
    # 添加国内镜像源
    mirrors = [
        "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/",
        "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/",
        "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/",
        "https://mirrors.ustc.edu.cn/anaconda/pkgs/main/",
        "https://mirrors.ustc.edu.cn/anaconda/pkgs/free/",
        "https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/"
    ]
    
    for mirror in mirrors:
        print(f"添加镜像源: {mirror}")
        success, stdout, stderr = run_command(f"conda config --add channels {mirror}")
        if success:
            print(f"✓ 成功添加 {mirror}")
        else:
            print(f"✗ 添加失败 {mirror}: {stderr}")
    
    # 设置conda使用国内镜像
    print("\n设置conda使用国内镜像...")
    run_command("conda config --set show_channel_urls yes")
    
    # 尝试安装pytorch-spline-conv
    print("\n尝试安装pytorch-spline-conv...")
    success, stdout, stderr = run_command("conda install pytorch-spline-conv=1.2.1 py38_torch_1.12.0_cu113 -c pyg -y")
    
    if success:
        print("✓ pytorch-spline-conv安装成功！")
        return True
    else:
        print("✗ conda安装失败，尝试使用pip...")
        return False

def install_with_pip():
    """使用pip安装预编译版本"""
    print("使用pip安装预编译版本...")
    
    # 尝试不同的CUDA版本
    cuda_versions = ["cu113", "cu116", "cu117", "cpu"]
    
    for cuda_ver in cuda_versions:
        url = f"https://data.pyg.org/whl/torch-1.12.0+{cuda_ver}/torch_spline_conv-1.2.1+pt112{cuda_ver}-cp38-cp38-win_amd64.whl"
        print(f"尝试安装: {cuda_ver}版本")
        
        success, stdout, stderr = run_command(f"pip install {url}")
        if success:
            print(f"✓ 成功安装 {cuda_ver}版本")
            return True
        else:
            print(f"✗ {cuda_ver}版本安装失败")
    
    # 最后尝试直接安装
    print("尝试直接安装...")
    success, stdout, stderr = run_command("pip install torch-spline-conv")
    if success:
        print("✓ 直接安装成功！")
        return True
    else:
        print("✗ 所有安装方法都失败了")
        return False

def main():
    """主函数"""
    print("torch-spline-conv安装工具")
    print("=" * 50)
    
    # 首先尝试修复conda
    if fix_conda_channels():
        print("\n安装完成！")
    else:
        # 如果conda失败，使用pip
        if install_with_pip():
            print("\n安装完成！")
        else:
            print("\n安装失败！请检查网络连接或手动下载wheel包")
    
    # 验证安装
    print("\n验证安装...")
    success, stdout, stderr = run_command("python -c \"import torch_spline_conv; print('torch-spline-conv版本:', torch_spline_conv.__version__)\"")
    if success:
        print("✓ 验证成功！")
    else:
        print("✗ 验证失败，但可能已成功安装")

if __name__ == "__main__":
    main()

