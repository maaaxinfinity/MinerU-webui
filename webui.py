import os
from zipfile import ZipFile
import json
import copy

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
import time
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
        output_dir: str = None
):
    """
    执行从 pdf 转换到 json、md 的过程，输出 md 和 json 文件到 pdf 文件所在的目录

    :param pdf_path: .pdf 文件的路径，可以是相对路径，也可以是绝对路径
    :param parse_method: 解析方法， 共 auto、ocr、txt 三种，默认 auto，如果效果不好，可以尝试 ocr
    :param model_json_path: 已经存在的模型数据文件，如果为空则使用内置模型，pdf 和 model_json 务必对应
    :param is_json_md_dump: 是否将解析后的数据写入到 .json 和 .md 文件中，默认 True，会将不同阶段的数据写入到不同的 .json 文件中（共3个.json文件），md内容会保存到 .md 文件中
    :param output_dir: 输出结果的目录地址，会生成一个以 pdf 文件名命名的文件夹并保存所有结果
    """
    progress(0, desc="正在启动任务...")
    logger.info("任务开始处理了")
    
    # 定义一个日志处理器函数，将日志信息添加到全局变量中
    def log_to_textbox(message):
        progress(1, desc=message)

    # 配置 loguru 日志记录
    logger.add(log_to_textbox, format="{time} {level} {message}", level="INFO")
    try:
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
                pipe = infer_result.pipe_ocr_mode(image_writer, debug_mode=True)
            elif parse_method == "txt":
                pipe = infer_result.pipe_txt_mode(image_writer, debug_mode=True)
            else:  # auto
                if ds.classify() == SupportedPdfParseMethod.TXT:
                    pipe = infer_result.pipe_txt_mode(image_writer, debug_mode=True)
                else:
                    pipe = infer_result.pipe_ocr_mode(image_writer, debug_mode=True)
        else:
            # 使用内置模型
            if model_config.__use_inside_model__:
                from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
                
                if parse_method == "auto":
                    if ds.classify() == SupportedPdfParseMethod.TXT:
                        infer_result = ds.apply(doc_analyze, ocr=False)
                        pipe = infer_result.pipe_txt_mode(image_writer, debug_mode=True)
                    else:
                        infer_result = ds.apply(doc_analyze, ocr=True)
                        pipe = infer_result.pipe_ocr_mode(image_writer, debug_mode=True)
                elif parse_method == "txt":
                    infer_result = ds.apply(doc_analyze, ocr=False)
                    pipe = infer_result.pipe_txt_mode(image_writer, debug_mode=True)
                elif parse_method == "ocr":
                    infer_result = ds.apply(doc_analyze, ocr=True)
                    pipe = infer_result.pipe_ocr_mode(image_writer, debug_mode=True)
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
        
        return [md_content, os.path.join(output_path, f"{pdf_name}.md")]
    except Exception as e:
        logger.exception(e)
        return None


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




def pdf_parse( pdf_path: str,
        progress=gr.Progress()):
    # 文件迁移到脚本目录的.temp
    file_name  = os.path.basename(pdf_path)
    pdf_name = file_name.split(".")[0]
    target_pdf_path = os.path.join(os.path.dirname(__file__), ".temp",file_name )
    # 复制文件到脚本目录的.temp
    with open(target_pdf_path, "wb") as f:
        f.write(open(pdf_path, "rb").read())
    # 开始解析
    [markdown_content,file_path] = pdf_parse_main(target_pdf_path,progress)
    # 替换markdown_content的所有图片，增加 /file=相对路径
    markdown_content = markdown_content.replace("![](", "![](/file=.temp/" + pdf_name+"/")
    return [markdown_content,file_path]

with gr.Blocks() as demo:
    with gr.Row(equal_height=True):
        with gr.Column():
            pdf_input = PDF(label="上传PDF文档",interactive=True)
            #pdf_input = gr.File(label="上传PDF文档", file_types=["pdf"])
            extract_button = gr.Button("开始抽取")

        with gr.Column():
            # 保存文件地址，用于后期打包
            base_path = gr.State("")
            export_button = gr.Button("打包下载")
            download_output = gr.File(label="导出")
            markdown_output = gr.Markdown(label="识别结果")

    extract_button.click(
        pdf_parse, 
        inputs=[pdf_input], 
        outputs=[markdown_output , base_path]
    )
    export_button.click(
        export_zip, 
        inputs=[base_path], 
        outputs=[download_output]
    )


logger.info(f"waiting for model init")
model_init = init_model()
logger.info(f"model_init: {model_init}")

demo.queue().launch(inbrowser=True,allowed_paths=["./temp"])