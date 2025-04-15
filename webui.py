import os
from zipfile import ZipFile
import json
import copy
import webbrowser
import threading
import socket
import time
import gradio as gr
import os
import subprocess
import re
import time
import json
import zipfile
import socket
import webbrowser
import sys
import logging
import argparse
import traceback
import platform
import shutil
from typing import List, Dict, Tuple, Any, Optional, Union
from datetime import datetime
from pdf2image import convert_from_path
import threading

from loguru import logger


from magic_pdf.data.dataset import Dataset
from magic_pdf.operators.models import InferenceResult
from magic_pdf.operators.pipes import PipeResult
from magic_pdf.data.data_reader_writer import DataWriter, FileBasedDataWriter
from magic_pdf.config.make_content_config import DropMode, MakeMode
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.config.constants import PARSE_TYPE_TXT, PARSE_TYPE_OCR
import magic_pdf.model as model_config

model_config.__use_inside_model__ = True
import gradio as gr
# 移除PDF组件导入
# from gradio_pdf import PDF
from pdf2image import convert_from_path
import threading

# 创建一个全局变量来存储日志信息
log_messages = []

def init_model():
    from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton
    try:
        model_manager = ModelSingleton()
        model_manager.get_model(False, False)
        logger.info(f"txt_model init final")
        model_manager.get_model(True, False)
        logger.info(f"ocr_model init final")
        return 0
    except Exception as e:
        logger.exception(e)
        return -1

gr.set_static_paths(paths=[".temp/","static/"])

def pdf_parse_main(
        pdf_path: str,
        progress=gr.Progress(),
        parse_method: str = 'auto',
        model_json_path: str = None,
        is_json_md_dump: bool = True,
        output_dir: str = None,
        formula_enable: bool = True,
        table_enable: bool = True,
        language: str = 'auto',
        end_pages: int = 1000,
        generate_preview: bool = True
):
    """
    执行从 pdf 转换到 json、md 的过程，输出 md 和 json 文件到 pdf 文件所在的目录

    :param pdf_path: .pdf 文件的路径，可以是相对路径，也可以是绝对路径
    :param parse_method: 解析方法， 共 auto、ocr、txt 三种，默认 auto，如果效果不好，可以尝试 ocr
    :param model_json_path: 已经存在的模型数据文件，如果为空则使用内置模型，pdf 和 model_json 务必对应
    :param is_json_md_dump: 是否将解析后的数据写入到 .json 和 .md 文件中，默认 True，会将不同阶段的数据写入到不同的 .json 文件中（共3个.json文件），md内容会保存到 .md 文件中
    :param output_dir: 输出结果的目录地址，会生成一个以 pdf 文件名命名的文件夹并保存所有结果
    :param formula_enable: 是否启用公式识别
    :param table_enable: 是否启用表格识别
    :param language: 识别语言，auto为自动识别
    :param end_pages: 最大处理页数
    :param generate_preview: 是否生成布局预览
    """
    # 添加对progress的检查
    if progress is None:
        def progress_dummy(value=0, desc=None):
            if desc:
                logger.info(desc)
        progress = progress_dummy

    progress(0, desc="正在启动任务...")
    logger.info("任务开始处理了")
    
    # 定义一个日志处理器函数，将日志信息添加到全局变量中
    def log_to_textbox(message):
        log_messages.append(message)
        progress(1, desc=message)

    # 配置 loguru 日志记录
    logger.add(log_to_textbox, format="{time} {level} {message}", level="INFO")
    try:
        start_time = time.time()
        
        pdf_name = os.path.basename(pdf_path).split(".")[0]
        pdf_path_parent = os.path.dirname(pdf_path)

        if output_dir:
            output_path = os.path.join(output_dir, pdf_name)
        else:
            output_path = os.path.join(pdf_path_parent, pdf_name)

        output_image_path = os.path.join(output_path, 'images')

        # 获取图片的父路径，为的是以相对路径保存到 .md 和 conent_list.json 文件中
        image_path_parent = os.path.basename(output_image_path)

        pdf_bytes = open(pdf_path, "rb").read()  # 读取 pdf 文件的二进制数据

        if model_json_path:
            # 读取已经被模型解析后的pdf文件的 json 原始数据，list 类型
            model_json = json.loads(open(model_json_path, "r", encoding="utf-8").read())
        else:
            model_json = []

        # 执行解析步骤
        image_writer = FileBasedDataWriter(output_image_path)
        md_writer = FileBasedDataWriter(output_path)

        # 创建Dataset对象
        from magic_pdf.data.dataset import PymuDocDataset
        ds = PymuDocDataset(pdf_bytes)

        # 选择解析方式
        infer_result = None
        pipe = None
        
        if model_json:
            # 使用已有的模型数据
            infer_result = InferenceResult(model_json, ds)
            if parse_method == "ocr":
                pipe = infer_result.pipe_ocr_mode(image_writer, debug_mode=True, end_page_id=end_pages-1)
            elif parse_method == "txt":
                pipe = infer_result.pipe_txt_mode(image_writer, debug_mode=True, end_page_id=end_pages-1)
            else:  # auto
                if ds.classify() == SupportedPdfParseMethod.TXT:
                    pipe = infer_result.pipe_txt_mode(image_writer, debug_mode=True, end_page_id=end_pages-1)
                else:
                    pipe = infer_result.pipe_ocr_mode(image_writer, debug_mode=True, end_page_id=end_pages-1)
        else:
            # 使用内置模型
            if model_config.__use_inside_model__:
                from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
                
                # 传递参数
                kwargs = {
                    "formula_enable": formula_enable,
                    "table_enable": table_enable,
                    "lang": language if language != "auto" else None
                }
                
                if parse_method == "auto":
                    if ds.classify() == SupportedPdfParseMethod.TXT:
                        infer_result = ds.apply(doc_analyze, ocr=False, **kwargs)
                        pipe = infer_result.pipe_txt_mode(image_writer, debug_mode=True, end_page_id=end_pages-1)
                    else:
                        infer_result = ds.apply(doc_analyze, ocr=True, **kwargs)
                        pipe = infer_result.pipe_ocr_mode(image_writer, debug_mode=True, end_page_id=end_pages-1)
                elif parse_method == "txt":
                    infer_result = ds.apply(doc_analyze, ocr=False, **kwargs)
                    pipe = infer_result.pipe_txt_mode(image_writer, debug_mode=True, end_page_id=end_pages-1)
                elif parse_method == "ocr":
                    infer_result = ds.apply(doc_analyze, ocr=True, **kwargs)
                    pipe = infer_result.pipe_ocr_mode(image_writer, debug_mode=True, end_page_id=end_pages-1)
            else:
                logger.error("内置模型不可用，需要提供模型数据")
                return None

        # 获取内容
        content_list = pipe.get_content_list(image_path_parent, drop_mode=DropMode.NONE)
        md_content = pipe.get_markdown(image_path_parent, drop_mode=DropMode.NONE)

        if is_json_md_dump:
            # 写入模型结果到 model.json
            if infer_result:
                orig_model_list = copy.deepcopy(infer_result.get_infer_res())
                md_writer.write_string(
                    f"model.json",
                    json.dumps(orig_model_list, ensure_ascii=False, indent=4)
                )

            # 写入中间结果到 middle.json
            if hasattr(pipe, '_pipe_res'):
                md_writer.write_string(
                    f"middle.json",
                    json.dumps(pipe._pipe_res, ensure_ascii=False, indent=4),
                )

            # text文本结果写入到 conent_list.json
            md_writer.write_string(
                f"content_list.json",
                json.dumps(content_list, ensure_ascii=False, indent=4),
            )

            # 写入结果到 .md 文件中
            md_writer.write_string(
                f"{pdf_name}.md",
                md_content
            )
        
        # 生成PDF布局预览
        layout_pdf_path = None
        if hasattr(pipe, 'draw_layout') and generate_preview:
            layout_pdf_path = os.path.join(output_path, f"{pdf_name}_layout.pdf")
            pipe.draw_layout(layout_pdf_path)
        
        # 统计处理时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        time_str = f"{minutes}分{seconds}秒"
        logger.info(f"处理完成！总耗时: {time_str}")
        
        return [md_content, os.path.join(output_path, f"{pdf_name}.md"), layout_pdf_path, f"处理完成！总耗时: {time_str}"]
    except Exception as e:
        logger.exception(e)
        return [None, None, None, f"处理失败: {str(e)}"]
 

# 定义一个函数来获取最新的日志信息
def get_logs():
    return "\n".join(log_messages)

def zip_files_and_dirs(files_to_zip, zip_file_path):
    # 创建一个ZipFile对象
    with ZipFile(zip_file_path, 'w') as zip_object:
        for item in files_to_zip:
            if os.path.isfile(item):
                # 如果是文件，直接添加到压缩文件
                zip_object.write(item, os.path.basename(item))
            elif os.path.isdir(item):
                # 如果是目录，遍历目录并添加到压缩文件
                for folder_name, sub_folders, file_names in os.walk(item):
                    for filename in file_names:
                        file_path = os.path.join(folder_name, filename)
                        # 使用相对路径来存储文件在压缩包中的路径
                        relative_path = os.path.relpath(file_path, item)
                        zip_object.write(file_path, os.path.join(os.path.basename(item), relative_path))
            else:
                print(f"警告: {item} 既不是文件也不是目录，跳过此项。")
def export_zip(base_path):
    # 获得父级路径
    parent_path = os.path.dirname(base_path)
    # 获得文件名
    file_name = os.path.basename(base_path)
    # image_dir
    images_dir = os.path.join(parent_path,  "images")
    # 压缩文件
    # 定义要压缩的文件和目标压缩文件的路径
    files_to_zip = [base_path, images_dir]
    zip_file_path =os.path.join(parent_path, f"{file_name}.zip")
    zip_files_and_dirs(files_to_zip, zip_file_path)

    return zip_file_path


def pdf_parse(pdf_path: str, progress=gr.Progress(), is_ocr=False, formula_enable=True, table_enable=True, language="auto", max_pages=1000, generate_preview=True):
    try:
        # 检查pdf_path是否有效
        if not pdf_path:
            logger.error("未选择PDF文件")
            return [None, None, None, "请先上传PDF文件"]
            
        # 确保.temp目录存在
        temp_dir = os.path.join(os.path.dirname(__file__), ".temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 记录文件信息，用于调试
        logger.info(f"处理文件: {pdf_path}")
        
        # 文件迁移到脚本目录的.temp
        file_name = os.path.basename(pdf_path)
        pdf_name = file_name.split(".")[0]
        target_pdf_path = os.path.join(os.path.dirname(__file__), ".temp", file_name)
        
        # 复制文件到脚本目录的.temp
        try:
            with open(target_pdf_path, "wb") as f:
                f.write(open(pdf_path, "rb").read())
            logger.info(f"文件已复制到: {target_pdf_path}")
        except Exception as e:
            logger.exception(f"复制文件失败: {str(e)}")
            return [None, None, None, f"读取PDF文件失败: {str(e)}"]
            
        # 开始解析
        parse_method = "ocr" if is_ocr else "auto"
        logger.info(f"开始解析，方法: {parse_method}, 语言: {language}, 最大页数: {max_pages}")
        
        [markdown_content, file_path, layout_pdf_path, status_msg] = pdf_parse_main(
            target_pdf_path, 
            progress, 
            parse_method=parse_method, 
            formula_enable=formula_enable, 
            table_enable=table_enable,
            language=language,
            end_pages=max_pages,
            generate_preview=generate_preview
        )
        
        # 替换markdown_content的所有图片，增加 /file=相对路径
        if markdown_content:
            # 改进图片路径替换逻辑，处理更多图片格式和路径情况
            import re
            # 1. 处理基本的Markdown图片引用 ![]()
            markdown_content = markdown_content.replace("![](", f"![](/file=.temp/{pdf_name}/")
            
            # 2. 处理包含alt文本的图片引用 ![alt](path)，但避免处理外部链接
            markdown_content = re.sub(r'!\[(.*?)\]\((?!http|https)([^)]+)\)', 
                                     r'![\1](/file=.temp/' + pdf_name + r'/\2)', 
                                     markdown_content)
            
            # 3. 处理HTML图片标签
            markdown_content = re.sub(r'<img\s+src=["\'](?!http|https)([^"\']+)["\']', 
                                     r'<img src="/file=.temp/' + pdf_name + r'/\1"', 
                                     markdown_content)
            
            logger.info("处理完成，已替换所有图片路径")
        else:
            logger.warning("处理完成，但markdown内容为空")
        
        # 如果存在layout_pdf_path，将其转换为图片
        preview_image_path = None
        if layout_pdf_path and os.path.exists(layout_pdf_path):
            try:
                # 创建预览图片存储目录
                preview_dir = os.path.join(temp_dir, "preview")
                os.makedirs(preview_dir, exist_ok=True)
                
                # 将PDF转换为多页图片
                preview_image_dir = os.path.join(preview_dir, f"{pdf_name}_layout_preview")
                os.makedirs(preview_image_dir, exist_ok=True)
                
                # 转换所有页面
                images = convert_from_path(layout_pdf_path)
                preview_images = []
                
                for i, img in enumerate(images):
                    img_path = os.path.join(preview_image_dir, f"page_{i+1}.png")
                    img.save(img_path, 'PNG')
                    preview_images.append(img_path)
                
                logger.info(f"创建了布局预览图: {len(preview_images)}页")
                
                # 返回第一页作为预览
                preview_image_path = preview_images[0] if preview_images else None
                
                # 将图片路径保存到layout_preview.json
                preview_json_path = os.path.join(preview_dir, f"{pdf_name}_layout_preview.json")
                with open(preview_json_path, 'w', encoding='utf-8') as f:
                    json.dump(preview_images, f, ensure_ascii=False)
                
            except Exception as e:
                logger.exception(f"创建布局预览图失败: {str(e)}")
                preview_image_path = None
            
        return [markdown_content, file_path, preview_image_path, status_msg]
        
    except Exception as e:
        logger.exception(f"PDF处理过程中发生错误: {str(e)}")
        return [None, None, None, f"处理出错: {str(e)}"]


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def open_browser(port):
    while not is_port_in_use(port):
        time.sleep(0.5)
    webbrowser.open(f'http://127.0.0.1:{port}')


# 定义支持的语言列表
latin_lang = [
    'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
    'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
    'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
    'sw', 'tl', 'tr', 'uz', 'vi', 'french', 'german'
]
arabic_lang = ['ar', 'fa', 'ug', 'ur']
cyrillic_lang = [
    'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
    'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
]
devanagari_lang = [
    'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
    'sa', 'bgc'
]
other_lang = ['ch', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']

all_lang = ['auto']
all_lang.extend([*other_lang, *latin_lang, *arabic_lang, *cyrillic_lang, *devanagari_lang])


# 设置方程式定界符配置
latex_delimiters = [
    {'left': '$$', 'right': '$$', 'display': True},
    {'left': '$', 'right': '$', 'display': False}
]

# 添加这两个新函数，用于处理多页显示
def get_preview_images(file_path):
    if not file_path or not os.path.exists(file_path):
        return []
    
    pdf_name = os.path.basename(file_path).split(".")[0]
    temp_dir = os.path.join(os.path.dirname(__file__), ".temp")
    preview_dir = os.path.join(temp_dir, "preview")
    preview_json_path = os.path.join(preview_dir, f"{pdf_name}_layout_preview.json")
    
    if os.path.exists(preview_json_path):
        try:
            with open(preview_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.exception(f"读取预览图JSON失败: {str(e)}")
    
    return []

def split_markdown_to_pages(markdown_text, page_size=1000):
    if not markdown_text:
        return []
    
    # 按照标题分割
    import re
    sections = re.split(r'(^#{1,6}\s.+$)', markdown_text, flags=re.MULTILINE)
    
    # 合并标题和内容
    pages = []
    current_page = ""
    current_size = 0
    
    for i in range(len(sections)):
        section = sections[i]
        if not section.strip():
            continue
            
        # 如果是标题
        if re.match(r'^#{1,6}\s.+$', section, flags=re.MULTILINE):
            # 如果当前页已经达到一定大小，开始新页
            if current_size >= page_size and current_page:
                pages.append(current_page)
                current_page = section
                current_size = len(section)
            else:
                current_page += section
                current_size += len(section)
                
            # 如果有下一节的内容，添加它
            if i + 1 < len(sections):
                current_page += sections[i+1]
                current_size += len(sections[i+1])
        # 如果不是标题且没有被前面处理过
        elif i > 0 and not re.match(r'^#{1,6}\s.+$', sections[i-1], flags=re.MULTILINE):
            current_page += section
            current_size += len(section)
            
            # 如果页面太大，拆分
            if current_size >= page_size * 2:
                pages.append(current_page)
                current_page = ""
                current_size = 0
    
    # 添加最后一页
    if current_page:
        pages.append(current_page)
        
    # 如果没有页面，将整个文档作为一页
    if not pages and markdown_text:
        pages.append(markdown_text)
        
    return pages


if __name__ == '__main__':
    # 设置日志输出到文件
    logger.add("mineru_webui.log", rotation="500 MB", level="DEBUG")
    logger.info("===== 应用启动 =====")
    
    port = 7860  # Gradio 默认端口
    
    # 启动打开浏览器的线程
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    logger.info(f"waiting for model init")
    model_init = init_model()
    logger.info(f"model_init: {model_init}")

    # 确保.temp目录存在
    temp_dir = os.path.join(os.path.dirname(__file__), ".temp")
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"确保.temp目录存在: {temp_dir}")
    
    # 读取header.html文件
    header_path = os.path.join(os.path.dirname(__file__), "header.html")
    with open(header_path, "r", encoding="utf-8") as file:
        header = file.read()

    with gr.Blocks(analytics_enabled=False, title="Mineru PDF处理", theme=gr.themes.Base()) as demo:
        try:
            # 添加HTML头部
            gr.HTML(header)
            
            with gr.Row():
                with gr.Column(variant='default', scale=5):
                    with gr.Tabs():
                        with gr.Tab('上传文件'):
                            # 使用简单的文件上传组件
                            pdf_input = gr.File(label="上传PDF文档", file_types=[".pdf"], interactive=True)
                            gr.HTML("<div style='margin-top:10px; margin-bottom:10px; color:#666;'>支持PDF文件格式</div>")
                            # 添加调试信息显示
                            pdf_debug = gr.Textbox(label="调试信息", visible=True)
                        
                        with gr.Tab('文档案例'):
                            example_root = os.path.join(os.path.dirname(__file__), 'examples')
                            if os.path.exists(example_root):
                                examples = [os.path.join(example_root, f) for f in os.listdir(example_root) if f.endswith('.pdf')]
                                if examples:
                                    gr.Examples(examples=examples, inputs=pdf_input)
                
                with gr.Column(variant='default', scale=5):
                    # 增加高级设置选项
                    max_pages = gr.Slider(1, 100000, 1000, step=1, label='最大转换页数')
                    with gr.Row():
                        language = gr.Dropdown(all_lang, label='语言', value='auto')
                    with gr.Row():
                        formula_enable = gr.Checkbox(label='启用公式识别', value=True)
                        is_ocr = gr.Checkbox(label='强制启用OCR识别', value=False)
                        table_enable = gr.Checkbox(label='启用表格识别', value=True)
                    with gr.Row():
                        generate_preview = gr.Checkbox(label='生成布局预览', value=True, info="更改此选项需要重新处理PDF")
                        preview_status = gr.HTML(value="<div></div>")
                    with gr.Row():
                        extract_button = gr.Button('开始抽取', variant='primary')
                        export_button = gr.Button('打包下载')
                        clear_button = gr.ClearButton(value='清空')
                
                with gr.Column(variant='default', scale=5):
                    status_output = gr.Textbox(label='处理状态', interactive=False)
                    download_output = gr.File(label='转换结果压缩包', interactive=False)

            with gr.Row():
                with gr.Column(variant='default', scale=5):
                    # 使用滑块组件显示布局预览
                    preview_slider = gr.Slider(minimum=1, maximum=1, step=1, value=1, label="布局预览页数")
                    preview_image = gr.Image(label='布局预览', interactive=False)
                    
                with gr.Column(variant='default', scale=5):
                    with gr.Tabs():
                        with gr.Tab('Markdown 渲染'):
                            # 添加Markdown分页控制
                            markdown_slider = gr.Slider(minimum=1, maximum=1, step=1, value=1, label="Markdown页数")
                            markdown_output = gr.Markdown(
                                label="识别结果", 
                                height=600, 
                                show_copy_button=True,
                                line_breaks=True,
                                sanitize_html=False,  # 允许HTML内容，使img标签可以正常渲染
                                elem_id="markdown-render-output"  # 添加元素ID方便CSS定制
                            )
                        with gr.Tab('Markdown 源码'):
                            # 添加Markdown源码分页控制
                            md_source_slider = gr.Slider(minimum=1, maximum=1, step=1, value=1, label="源码页数")
                            md_text = gr.TextArea(
                                label="源代码", 
                                lines=40, 
                                show_copy_button=True
                            )
            
            # 保存文件地址，用于后期打包
            base_path = gr.State("")
            # 保存Markdown分页内容
            markdown_pages = gr.State([])
            # 保存预览图片路径
            preview_images = gr.State([])
            
            # 添加PDF输入变化的事件处理
            pdf_input.change(
                lambda x: f"选择的文件: {str(x)[:100] if x else '无文件'}", 
                inputs=[pdf_input], 
                outputs=[pdf_debug]
            )
            
            # 处理布局预览滑块变化
            def update_preview(page_num, image_paths):
                if not image_paths or page_num <= 0 or page_num > len(image_paths):
                    return None
                return image_paths[page_num-1]
            
            # 处理Markdown渲染滑块变化
            def update_markdown(page_num, pages):
                if not pages or page_num <= 0 or page_num > len(pages):
                    return ""
                return pages[page_num-1]
            
            # 添加新函数：同步所有滑块
            def sync_all_sliders(page_num, preview_images, markdown_pages):
                """同步所有滑块到相同的页码"""
                # 确保页码在有效范围内
                if not preview_images or not markdown_pages:
                    return [None, "", "", page_num, page_num, page_num]
                
                preview_max = len(preview_images)
                md_max = len(markdown_pages)
                
                # 确保页码在有效范围内
                if page_num <= 0:
                    page_num = 1
                elif page_num > min(preview_max, md_max):
                    page_num = min(preview_max, md_max)
                
                # 获取相应页的预览图和Markdown内容
                preview = update_preview(page_num, preview_images)
                markdown = update_markdown(page_num, markdown_pages)
                
                # 返回所有更新的值
                return [preview, markdown, markdown, page_num, page_num, page_num]
            
            # 将原来单独的滑块事件处理绑定替换为新的同步函数绑定
            preview_slider.change(
                sync_all_sliders,
                inputs=[preview_slider, preview_images, markdown_pages],
                outputs=[preview_image, markdown_output, md_text, preview_slider, markdown_slider, md_source_slider]
            )
            
            markdown_slider.change(
                sync_all_sliders,
                inputs=[markdown_slider, preview_images, markdown_pages],
                outputs=[preview_image, markdown_output, md_text, preview_slider, markdown_slider, md_source_slider]
            )
            
            md_source_slider.change(
                sync_all_sliders,
                inputs=[md_source_slider, preview_images, markdown_pages],
                outputs=[preview_image, markdown_output, md_text, preview_slider, markdown_slider, md_source_slider]
            )
            
            # 添加布局预览开关状态变化的处理函数
            def update_preview_status(generate_preview_value):
                if generate_preview_value:
                    return "<div style='color:green;margin-top:5px;'>布局预览已启用</div>"
                else:
                    return "<div style='color:orange;margin-top:5px;'>布局预览已禁用，如需开启请先勾选再重新处理</div>"

            generate_preview.change(
                update_preview_status,
                inputs=[generate_preview],
                outputs=[preview_status]
            )
            
            # 定义事件处理函数
            def handle_extract(pdf_file, is_ocr, formula_enable, table_enable, language, max_pages, generate_preview):
                # 先显示调试信息
                if pdf_file is None:
                    logger.warning("未选择文件")
                    return [None, None, None, "请先上传PDF文件", [], []]
                
                # 检查文件是否有效
                try:
                    if isinstance(pdf_file, list) and len(pdf_file) > 0:
                        # 如果传入的是文件列表，取第一个
                        pdf_path = pdf_file[0].name
                    elif hasattr(pdf_file, 'name'):
                        # 如果传入的是单个文件对象
                        pdf_path = pdf_file.name
                    else:
                        # 如果传入的是字符串路径
                        pdf_path = pdf_file
                    
                    # 记录当前处理的参数
                    logger.info(f"处理参数: 文件={pdf_path}, OCR={is_ocr}, 公式={formula_enable}, 表格={table_enable}, 语言={language}, 最大页数={max_pages}, 生成预览={generate_preview}")
                    # 调用PDF处理函数
                    [md_content, file_path, preview_image_path, status_msg] = pdf_parse(pdf_path, gr.Progress(), is_ocr, formula_enable, table_enable, language, max_pages, generate_preview)
                    
                    # 获取所有预览图片
                    preview_images_list = get_preview_images(file_path)
                    
                    # 分割Markdown内容为页面
                    md_pages = split_markdown_to_pages(md_content)
                    
                    # 更新滑块最大值
                    preview_slider_value = gr.Slider(minimum=1, maximum=max(1, len(preview_images_list)), step=1, value=1, label="布局预览页数")
                    markdown_slider_value = gr.Slider(minimum=1, maximum=max(1, len(md_pages)), step=1, value=1, label="Markdown页数")
                    md_source_slider_value = gr.Slider(minimum=1, maximum=max(1, len(md_pages)), step=1, value=1, label="源码页数")
                    
                    # 返回第一页的预览和Markdown内容
                    first_preview = preview_image_path
                    first_markdown = md_pages[0] if md_pages else ""
                    
                    return [
                        first_markdown, 
                        file_path, 
                        first_preview, 
                        status_msg, 
                        md_pages, 
                        preview_images_list, 
                        preview_slider_value, 
                        markdown_slider_value, 
                        md_source_slider_value,
                        first_markdown
                    ]
                except Exception as e:
                    logger.exception(f"处理文件出错: {str(e)}")
                    return [None, None, None, f"处理文件出错: {str(e)}", [], [], gr.Slider(1), gr.Slider(1), gr.Slider(1), None]
            
            # 绑定提取按钮点击事件
            extract_button.click(
                handle_extract,
                inputs=[pdf_input, is_ocr, formula_enable, table_enable, language, max_pages, generate_preview], 
                outputs=[
                    markdown_output, 
                    base_path, 
                    preview_image, 
                    status_output, 
                    markdown_pages, 
                    preview_images, 
                    preview_slider,
                    markdown_slider,
                    md_source_slider,
                    md_text
                ]
            )
            
            export_button.click(
                export_zip, 
                inputs=[base_path], 
                outputs=[download_output]
            )

            # 添加清空按钮
            clear_button.add([
                pdf_input, 
                markdown_output, 
                md_text, 
                download_output, 
                preview_image, 
                status_output, 
                pdf_debug, 
                markdown_pages, 
                preview_images,
                preview_slider,
                markdown_slider,
                md_source_slider,
                preview_status
            ])

            # 在合适位置添加这段代码，用于为Markdown渲染添加CSS样式
            demo.load(lambda: gr.HTML("""
            <style>
                /* 增强Markdown中图片的显示 */
                #markdown-render-output img {
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 10px auto;
                    border: 1px solid #eee;
                    border-radius: 5px;
                }
                /* 增强公式显示 */
                .katex-display {
                    overflow-x: auto;
                    overflow-y: hidden;
                    padding: 5px 0;
                }
            </style>
            """), None, None)

        except Exception as e:
            logger.exception(f"界面创建出错: {str(e)}")
            gr.HTML(f"<div style='color:red'>应用初始化失败: {str(e)}</div>")

    # 启动Gradio界面
    logger.info("启动Gradio界面")
    try:
        demo.queue().launch(
            server_name='127.0.0.1', 
            server_port=port, 
            share=False, 
            allowed_paths=[".temp/", "static/", "examples/"],  # 允许访问.temp和其他目录下的文件
            debug=True,
            show_error=True,  # 显示错误信息，帮助调试
        )
    except Exception as e:
        logger.exception(f"Gradio启动失败: {str(e)}")