# MinerU WebUI

MinerU WebUI是一个基于Gradio的图形用户界面，用于MinerU项目的PDF处理和识别功能。

## 项目更新说明

**重要更新**: 本代码已适配最新版MinerU API结构。如果您遇到以下错误：

```
ImportError: cannot import name 'DiskReaderWriter' from 'magic_pdf.data.data_reader_writer'
```

请确保使用最新版代码，我们已经修复了此问题。

## 特性

- 简单易用的Web界面
- 支持PDF文档上传和处理
- 多种解析模式：自动、文本和OCR
- 结果展示和导出功能

## 安装

### 1. 安装MinerU项目

首先必须安装MinerU项目。请按照以下步骤操作：

```bash
# 克隆MinerU仓库（如果还没有）
git clone https://github.com/your-repo/MinerU.git

# 进入MinerU目录并安装
cd MinerU
pip install -e .
```

### 2. 安装WebUI

```bash
# 克隆WebUI仓库
git clone https://github.com/your-repo/MinerU-webui.git

# 进入WebUI目录并安装依赖
cd MinerU-webui
pip install -r requirements.txt

# 创建临时目录（用于存储上传的文件）
mkdir -p .temp
```

## 使用方法

启动WebUI：

```bash
python webui.py
```

然后在浏览器中打开显示的链接（默认为 http://127.0.0.1:7860）

### 使用步骤

1. 在界面上传PDF文件
2. 点击"开始抽取"按钮
3. 等待处理完成，查看抽取结果
4. 如需下载处理后的文件，点击"打包下载"按钮

## API服务

本项目还提供了API服务，可以通过以下命令启动：

```bash
python api.py
```

API服务默认运行在http://127.0.0.1:8080，提供以下接口：

- `POST /`: 上传PDF文件并创建解析任务
- `GET /task/{task_id}`: 查询任务状态
- `GET /pack/{task_id}`: 打包下载解析结果

详细API文档可访问: http://127.0.0.1:8080/docs

## 排错指南

### 常见错误

1. **导入错误**:

   - `ImportError: No module named 'magic_pdf.pipe'`
   - `ImportError: cannot import name 'DiskReaderWriter'`

   **解决方案**: 这是因为MinerU项目API结构更新导致的，确保使用最新版本的WebUI代码。

2. **模型初始化错误**:
   
   确保按照MinerU项目的指南正确下载和配置模型文件。

## 更新记录

- 2023-07-01: 初始版本
- 2023-12-01: 适配最新MinerU API结构，修复了导入路径问题
- 2024-06-01: 修复DiskReaderWriter导入错误，全面更新代码适配新版API

## 注意事项

- 本项目需要依赖MinerU项目，确保已正确安装
- 处理大型PDF文件可能需要较长时间和较大内存
- 本WebUI主要针对中文PDF设计，其他语言的支持可能有限

<div align="center" xmlns="http://www.w3.org/1999/html">
<!-- logo -->
<p align="center">
  <img src="doc/mineru.png" width="300px" style="vertical-align:middle;">
</p> 
<!-- hot link --> 
<p align="center">
<a href="https://github.com/opendatalab/PDF-Extract-Kit">PDF-Extract-Kit: High-Quality PDF Extraction Toolkit</a>🔥🔥🔥
</p>
</div>

## 项目简介
MinerU是一款开源的高质量PDF解析工具，基于深度学习技术，可自动提取PDF文档中的文字、表格、图片、公式等内容，并提供丰富的分析、统计、搜索等功能。
本项目为其提供一个简化版本的WebUI，方便用户上传PDF文件，并实时展示提取结果。

![页面截图](doc/preview.png)


## 🚀推荐 `MinerU-PDFScanner` 🌟
欢迎大家使用配套的新程序`MinerU-PDFScanner`, 可以将本程序作为后端调用，支持多任务并行处理，支持历史数据查看，支持导出包

<div align="center" xmlns="http://www.w3.org/1999/html">
<!-- logo -->
<p align="center">
  <img src="doc/mineru.png" width="300px" style="vertical-align:middle;">
</p> 
<!-- hot link --> 
<p align="center">
👏👏👏👏👏👏<a href="https://liuhuapiaoyuan.github.io/MinerU-PDFScanner/">MinerU-PDFScanner</a>🔥🔥🔥
</p>
</div>


![页面截图](doc/image.png)

## 说明：有条件的，用`gpu`加速，提升OCR解析速度
1. 卸载 `pip uninstall paddlepaddle`
2. 安装cuda版本对应的paddle 

## 更新日志
### 2024.10.11
- 支持API接口 👏👏👏👏
- 接口支持异步处理，并发处理🔥🔥🔥🔥
- 接口文档`http://127.0.0.1:8080/docs#/`

## 1. 第一次运行前的配置
在仓库根目录可以获得 [magic-pdf.template.json](doc/magic-pdf.template.json) 配置模版文件
> ❗️务必执行以下命令将配置文件拷贝到【用户目录】下，否则程序将无法运行
>  
> windows用户目录为 "C:\Users\用户名"
```powershell
(New-Object System.Net.WebClient).DownloadFile('https://gitee.com/myhloli/MinerU/raw/master/magic-pdf.template.json', 'magic-pdf.template.json')
cp magic-pdf.template.json ~/magic-pdf.json
```

在用户目录中找到magic-pdf.json文件并配置"models-dir"为[5. 下载模型](#5-下载模型)中下载的模型权重文件所在目录
> ❗️务必正确配置模型权重文件所在目录的【绝对路径】，否则会因为找不到模型文件而导致程序无法运行
> 
> windows系统中此路径应包含盘符，且需把路径中所有的`"\"`替换为`"/"`,否则会因为转义原因导致json文件语法错误。
> 
> 例如：模型放在D盘根目录的models目录，则model-dir的值应为"D:/models"
```json
{
  "models-dir": "/tmp/models"
}
```


## 2. 测试CUDA加速
如果您的显卡显存大于等于8G，可以进行以下流程，测试CUDA解析加速效果

**1.覆盖安装支持cuda的torch和torchvision**
```bash
pip install --force-reinstall torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
```
> ❗️务必在命令中指定以下版本
> ```bash
> torch==2.3.1 torchvision==0.18.1 
> ```
> 这是我们支持的最高版本，如果不指定版本会自动安装更高版本导致程序无法运行

**2.修改【用户目录】中配置文件magic-pdf.json中"device-mode"的值**
```json
{
  "device-mode":"cuda"
}
```

> 提示：CUDA加速是否生效可以根据log中输出的各个阶段cost耗时来简单判断，通常情况下，`layout detection cost` 和 `mfr time` 应提速10倍以上。


## 案例PDF可以选择 `examples/small_ocr.pdf`

## 如何下载模型权重文件

详细参考 [如何下载模型文件](https://github.com/opendatalab/MinerU/blob/master/docs/how_to_download_models_zh_cn.md)
> ❗️模型下载后请务必检查模型文件是否下载完整
> 
> 请检查目录下的模型文件大小与网页上描述是否一致，如果可以的话，最好通过sha256校验模型是否下载完整
> 下载完成后，建议放入目录`models`，并且修改配置文件`magic-pdf.json`中的`models-dir`为模型文件所在目录的绝对路径。

## 启动项目
```bash
conda create -n MinerU python=3.10
conda activate MinerU
pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
python webui.py
```

> 启动成功后，浏览器打开 http://localhost:7860/ 即可访问项目主页。

## TODO：
- [ ] 支持进度查看
- [ ] 支持多文件上传，批处理

## 常见问题

### 文件上传失败：目录不存在错误

如果遇到以下错误：
```
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/.temp/your_file.pdf'
```

这是因为程序无法找到`.temp`目录。解决方法：

1. 确保在MinerU-webui目录下创建了`.temp`目录：
```bash
mkdir -p .temp
```

2. 如果使用的是最新版代码，程序会自动创建此目录，无需手动创建

### 使用API服务时的目录问题

如果使用API服务，确保`uploads`目录也存在：
```bash
mkdir -p uploads
```

最新版的代码已经处理了目录自动创建的问题。