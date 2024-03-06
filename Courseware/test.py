import os

def convert_pptx_to_pdf(input_folder, output_folder):
    # 遍历输入文件夹中的所有PPTX文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".ppt") or filename.endswith(".pptx"):
            # 构建输入和输出文件的完整路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".pdf")

            # 使用unoconv将PPTX转换为PDF
            os.system(f"unoconv -f pdf '{input_path}'")

            # 将生成的PDF文件移动到输出文件夹
            print(f"Converted {filename} to PDF")

# 设置输入和输出文件夹的路径
input_folder = output_folder = os.path.dirname(__file__)

# 执行转换
convert_pptx_to_pdf(input_folder, output_folder)
