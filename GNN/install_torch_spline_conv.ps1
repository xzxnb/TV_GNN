# PowerShell脚本：安装torch-spline-conv预编译版本
Write-Host "正在安装torch-spline-conv预编译版本..." -ForegroundColor Green

# 检查Python是否安装
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python版本: $pythonVersion" -ForegroundColor Cyan
} catch {
    Write-Host "错误: 未找到Python，请先安装Python" -ForegroundColor Red
    Read-Host "按任意键退出"
    exit 1
}

# 检查PyTorch是否安装
try {
    $pytorchInfo = python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"无\"}')" 2>&1
    Write-Host $pytorchInfo -ForegroundColor Cyan
} catch {
    Write-Host "错误: 未找到PyTorch，请先安装PyTorch" -ForegroundColor Red
    Read-Host "按任意键退出"
    exit 1
}

# 获取PyTorch版本和CUDA信息
$pytorchVersion = python -c "import torch; print(torch.__version__)" 2>&1
$cudaAvailable = python -c "import torch; print(torch.cuda.is_available())" 2>&1

if ($cudaAvailable -eq "True") {
    $cudaVersion = python -c "import torch; print(torch.version.cuda)" 2>&1
    $cudaSuffix = "cu$cudaVersion"
    Write-Host "使用CUDA版本安装 (CUDA $cudaVersion)" -ForegroundColor Yellow
} else {
    $cudaSuffix = "cpu"
    Write-Host "使用CPU版本安装" -ForegroundColor Yellow
}

# 构建wheel包URL
$wheelUrl = "https://data.pyg.org/whl/torch-$pytorchVersion+$cudaSuffix/torch_spline_conv-1.2.1+pt112$cudaSuffix-cp38-cp38-win_amd64.whl"

Write-Host "安装URL: $wheelUrl" -ForegroundColor Cyan
Write-Host ""

# 尝试安装预编译版本
Write-Host "正在安装预编译版本..." -ForegroundColor Green
try {
    python -m pip install $wheelUrl
    if ($LASTEXITCODE -eq 0) {
        Write-Host "预编译版本安装成功！" -ForegroundColor Green
    } else {
        throw "安装失败"
    }
} catch {
    Write-Host "预编译版本安装失败，尝试直接安装..." -ForegroundColor Yellow
    try {
        python -m pip install torch-spline-conv
        if ($LASTEXITCODE -eq 0) {
            Write-Host "直接安装成功！" -ForegroundColor Green
        } else {
            throw "直接安装也失败"
        }
    } catch {
        Write-Host "安装失败！请检查网络连接或手动下载wheel包" -ForegroundColor Red
        Write-Host "您可以尝试手动下载并安装以下文件:" -ForegroundColor Yellow
        Write-Host $wheelUrl -ForegroundColor Cyan
        Read-Host "按任意键退出"
        exit 1
    }
}

Write-Host ""
Write-Host "正在验证安装..." -ForegroundColor Green

# 验证安装
try {
    $version = python -c "import torch_spline_conv; print(torch_spline_conv.__version__)" 2>&1
    Write-Host "torch-spline-conv安装成功！版本: $version" -ForegroundColor Green
} catch {
    Write-Host "警告: 无法验证安装，但可能已成功安装" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "安装完成！" -ForegroundColor Green
Read-Host "按任意键退出"

