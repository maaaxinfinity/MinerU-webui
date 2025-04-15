# MinerU WebUI

MinerU WebUI是一个基于Gradio的图形用户界面，用于PDF文档的智能解析和内容提取。它封装了MinerU项目的PDF处理和识别功能，提供简单易用的操作界面。

Fork自原项目 https://github.com/liuhuapiaoyuan/MinerU-webui

## 功能特点

- 界面友好：基于Gradio构建的直观Web界面
- PDF智能解析：支持文本、表格、图片、公式等内容提取
- 多种解析模式：支持自动、文本和OCR三种解析模式
- 丰富语言支持：支持多种语言识别，包括中文、英文等
- 分页浏览：支持布局预览和Markdown内容的分页显示
- 结果导出：支持一键打包下载解析结果

## 安装说明

### 1. 安装MinerU项目

首先必须安装MinerU项目。请访问MinerU的官方仓库下载模型：
https://github.com/opendatalab/MinerU

### 2. 安装WebUI

```bash
# 克隆WebUI仓库
git clone https://github.com/your-repo/MinerU-webui.git

# 进入WebUI目录并安装依赖
cd MinerU-webui
pip install -r requirements.txt

# 创建必要的目录
mkdir -p .temp
```

确保拷贝magic-pdf.json配置文件到用户目录：

```bash
cp ~/magic-pdf.json ./
```

然后编辑配置文件，设置正确的模型目录路径。


## 使用方法

启动WebUI：

```bash
python webui.py
```

然后在浏览器中打开显示的链接（默认为 http://127.0.0.1:7860）

### 使用步骤

1. 在界面上传PDF文件
2. 选择适当的解析参数：
   - 语言：选择PDF文档的主要语言（默认为自动检测）
   - 最大转换页数：限制处理页数
   - 其他选项：公式识别、表格识别、OCR识别等
3. 点击"开始抽取"按钮
4. 等待处理完成，查看抽取结果
   - 可使用滑块在布局预览中查看不同页面
   - 可使用滑块在Markdown渲染和源码标签页中查看不同部分
5. 如需下载处理后的文件，点击"打包下载"按钮

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

1. **PDF2Image相关错误**:
   
   如果遇到与pdf2image相关的错误，请确保正确安装了Poppler
   
   - Windows用户确保将Poppler的bin目录添加到PATH
   - Linux/Mac用户确保安装了poppler-utils

2. **模型初始化错误**:
   
   确保按照MinerU项目的指南正确下载和配置模型文件，并在magic-pdf.json中正确设置路径。

3. **目录不存在错误**:

   如果遇到以下错误：
   ```
   FileNotFoundError: [Errno 2] No such file or directory: '/path/to/.temp/your_file.pdf'
   ```
   
   请确保创建了`.temp`目录：
   ```bash
   mkdir -p .temp
   ```

## GPU加速

如果您的显卡显存大于等于8G，可以启用CUDA加速：

1. 覆盖安装支持CUDA的PyTorch：
   ```bash
   pip install --force-reinstall torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
   ```

2. 修改用户目录中的magic-pdf.json配置文件：
   ```json
   {
     "device-mode":"cuda"
   }
   ```

## 项目简介

MinerU是一款开源的高质量PDF解析工具，基于深度学习技术，可自动提取PDF文档中的文字、表格、图片、公式等内容，并提供丰富的分析、统计、搜索等功能。
本项目为其提供一个简化版本的WebUI，方便用户上传PDF文件，并实时展示提取结果。

![页面截图](doc/preview.png)

## 推荐 `MinerU-PDFScanner`

欢迎大家使用配套的新程序`MinerU-PDFScanner`, 可以将本程序作为后端调用，支持多任务并行处理，支持历史数据查看，支持导出包：

<div align="center" xmlns="http://www.w3.org/1999/html">
<p align="center">
  <img src="doc/mineru.png" width="300px" style="vertical-align:middle;">
</p> 
<p align="center">
<a href="https://liuhuapiaoyuan.github.io/MinerU-PDFScanner/">MinerU-PDFScanner</a>
</p>
</div>

![页面截图](doc/image.png)

## 更新日志

- 2024.04.15: 添加布局预览和Markdown内容的分页功能(Fork)
- 2024.xx.xx: 修复进度条处理问题
- 2024.10.11: 
  - 支持API接口
  - 接口支持异步处理，并发处理
  - 接口文档`http://127.0.0.1:8080/docs#/`


## 相关项目

<p align="center">
<a href="https://github.com/opendatalab/PDF-Extract-Kit">PDF-Extract-Kit: High-Quality PDF Extraction Toolkit</a>
</p>