import os
from zipfile import ZipFile
import json
import copy
import webbrowser
import threading
import socket
import time

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
from gradio_pdf import PDF
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
        end_pages: int = 1000
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
    """
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
        if hasattr(pipe, 'draw_layout'):
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


def pdf_parse(pdf_path: str, progress=gr.Progress(), is_ocr=False, formula_enable=True, table_enable=True, language="auto", max_pages=1000):
    # 确保.temp目录存在
    temp_dir = os.path.join(os.path.dirname(__file__), ".temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 文件迁移到脚本目录的.temp
    file_name = os.path.basename(pdf_path)
    pdf_name = file_name.split(".")[0]
    target_pdf_path = os.path.join(os.path.dirname(__file__), ".temp", file_name)
    # 复制文件到脚本目录的.temp
    with open(target_pdf_path, "wb") as f:
        f.write(open(pdf_path, "rb").read())
    # 开始解析
    parse_method = "ocr" if is_ocr else "auto"
    [markdown_content, file_path, layout_pdf_path, status_msg] = pdf_parse_main(
        target_pdf_path, 
        progress, 
        parse_method=parse_method, 
        formula_enable=formula_enable, 
        table_enable=table_enable,
        language=language,
        end_pages=max_pages
    )
    
    # 替换markdown_content的所有图片，增加 /file=相对路径
    if markdown_content:
        markdown_content = markdown_content.replace("![](", "![](/file=.temp/" + pdf_name+"/")
        
    return [markdown_content, file_path, layout_pdf_path, status_msg]


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


if __name__ == '__main__':
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

    with gr.Blocks() as demo:
        # 添加HTML头部
        gr.HTML(header)
        
        with gr.Row():
            with gr.Column(variant='panel', scale=5):
                with gr.Tabs():
                    with gr.Tab('上传文件'):
                        pdf_input = PDF(label="上传PDF文档", interactive=True)
                        gr.HTML("<div style='margin-top:10px; margin-bottom:10px; color:#666;'>支持PDF文件格式</div>")
                    
                    with gr.Tab('文档案例'):
                        example_root = os.path.join(os.path.dirname(__file__), 'examples')
                        if os.path.exists(example_root):
                            examples = [os.path.join(example_root, f) for f in os.listdir(example_root) if f.endswith('.pdf')]
                            if examples:
                                gr.Examples(examples=examples, inputs=pdf_input)
            
            with gr.Column(variant='panel', scale=5):
                # 增加高级设置选项
                max_pages = gr.Slider(1, 100000, 1000, step=1, label='最大转换页数')
                with gr.Row():
                    language = gr.Dropdown(all_lang, label='语言', value='auto')
                with gr.Row():
                    formula_enable = gr.Checkbox(label='启用公式识别', value=True)
                    is_ocr = gr.Checkbox(label='强制启用OCR识别', value=False)
                    table_enable = gr.Checkbox(label='启用表格识别', value=True)
                with gr.Row():
                    extract_button = gr.Button('开始抽取', variant='primary')
                    export_button = gr.Button('打包下载')
                    clear_button = gr.ClearButton(value='清空')
            
            with gr.Column(variant='panel', scale=5):
                with gr.Tabs():
                    download_output = gr.File(label='转换结果压缩包', interactive=False, height=50)
                    status_output = gr.Textbox(label='处理状态', interactive=False)

        with gr.Row():
            with gr.Column(variant='panel', scale=5):
                # 这里显示PDF预览
                preview_pdf = gr.PDF(label='布局预览', interactive=False, visible=True, height=800)
                
            with gr.Column(variant='panel', scale=5):
                with gr.Tabs():
                    with gr.Tab('Markdown 渲染'):
                        markdown_output = gr.Markdown(
                            label="识别结果", 
                            height=800, 
                            show_copy_button=True,
                            latex_delimiters=latex_delimiters,
                            line_breaks=True
                        )
                    with gr.Tab('Markdown 源码'):
                        md_text = gr.TextArea(
                            label="源代码", 
                            lines=60, 
                            show_copy_button=True
                        )
        
        # 保存文件地址，用于后期打包
        base_path = gr.State("")
        
        # 绑定事件
        extract_button.click(
            pdf_parse, 
            inputs=[pdf_input, is_ocr, formula_enable, table_enable, language, max_pages], 
            outputs=[markdown_output, base_path, preview_pdf, status_output]
        )
        
        export_button.click(
            export_zip, 
            inputs=[base_path], 
            outputs=[download_output]
        )
        
        # 将markdown_output添加到md_text用于显示源码
        markdown_output.change(
            lambda x: x, 
            inputs=[markdown_output], 
            outputs=[md_text]
        )
        
        # 添加清空按钮
        clear_button.add([pdf_input, markdown_output, md_text, download_output, preview_pdf, status_output])

    demo.queue().launch(server_name='127.0.0.1', server_port=port, share=True, allowed_paths=[".temp/", "static/"])