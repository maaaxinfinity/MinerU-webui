import os
import json
import copy

from loguru import logger

# 更新导入路径以适配新版MinerU
from magic_pdf.data.dataset import Dataset, PymuDocDataset 
from magic_pdf.operators.models import InferenceResult
from magic_pdf.operators.pipes import PipeResult
from magic_pdf.data.data_reader_writer import DataWriter, FileBasedDataWriter
from magic_pdf.config.make_content_config import DropMode, MakeMode
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.config.constants import PARSE_TYPE_TXT, PARSE_TYPE_OCR
import magic_pdf.model as model_config

model_config.__use_inside_model__ = True

# todo: 设备类型选择 （？）

# 更新json_md_dump函数
def json_md_dump(
        infer_result,
        pipe,
        md_writer,
        pdf_name,
        content_list,
        md_content,
):
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


def pdf_parse_main(
        pdf_path: str,
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
            json_md_dump(infer_result, pipe, md_writer, pdf_name, content_list, md_content)

        return [md_content, os.path.join(output_path, f"{pdf_name}.md")]

    except Exception as e:
        logger.exception(e)
        return None


# 测试
if __name__ == '__main__':
    pdf_path = r"E:\AI_TOOLS\MinerU\ddd.pdf"
    pdf_parse_main(pdf_path)