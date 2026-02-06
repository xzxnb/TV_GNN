@echo off
echo 正在安装torch-spline-conv预编译版本...

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python
    pause
    exit /b 1
)

REM 检查PyTorch是否安装
python -c "import torch; print('PyTorch版本:', torch.__version__)" >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到PyTorch，请先安装PyTorch
    pause
    exit /b 1
)

echo 正在检测PyTorch版本和CUDA支持...

REM 获取PyTorch版本
for /f "tokens=*" %%i in ('python -c "import torch; print(torch.__version__)"') do set PYTORCH_VERSION=%%i

REM 检查CUDA支持
python -c "import torch; print('CUDA:', torch.cuda.is_available())" > temp_cuda.txt
findstr "True" temp_cuda.txt >nul
if errorlevel 1 (
    echo 使用CPU版本安装...
    set CUDA_SUFFIX=cpu
    set CUDA_VERSION=
) else (
    echo 使用CUDA版本安装...
    for /f "tokens=*" %%i in ('python -c "import torch; print(torch.version.cuda)"') do set CUDA_VERSION=%%i
    set CUDA_SUFFIX=cu!CUDA_VERSION!
)

REM 清理临时文件
del temp_cuda.txt >nul 2>&1

echo PyTorch版本: %PYTORCH_VERSION%
echo CUDA后缀: %CUDA_SUFFIX%

REM 构建安装URL
set WHEEL_URL=https://data.pyg.org/whl/torch-%PYTORCH_VERSION%+%CUDA_SUFFIX%/torch_spline_conv-1.2.1+pt112%CUDA_SUFFIX%-cp38-cp38-win_amd64.whl

echo 安装URL: %WHEEL_URL%
echo.

REM 尝试安装预编译版本
echo 正在安装预编译版本...
pip install %WHEEL_URL%

if errorlevel 1 (
    echo 预编译版本安装失败，尝试直接安装...
    pip install torch-spline-conv
    if errorlevel 1 (
        echo 安装失败！请检查网络连接或手动下载wheel包
        pause
        exit /b 1
    )
)

echo.
echo torch-spline-conv安装完成！
echo 正在验证安装...

python -c "import torch_spline_conv; print('torch-spline-conv安装成功！版本:', torch_spline_conv.__version__)"

pause

