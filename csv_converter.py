#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV转换脚本
将output的csv格式转换为template_import.csv格式

功能：
- 将OCR识别结果CSV转换为题库导入格式
- 支持单选题、多选题、判断题
- 自动查找最新的output CSV文件
- 交互式选择文件夹并显示创建日期
- 使用实际解析内容（不再是固定字符串）

使用方法：
1. 交互式模式（默认）：python csv_converter.py
   - 显示所有可用文件夹，让用户手动选择（推荐）

2. 自动模式：python csv_converter.py --auto
   - 自动查找output目录中最新的ocr_results.csv文件

3. 指定文件：python csv_converter.py <输入文件路径>
   - 手动指定要转换的CSV文件路径

输入格式（output CSV）：
序号,题型,题干,选项A,选项B,选项C,选项D,选项E,答案,解析,已修正,修正说明

输出格式（template_import.csv）：
题干,答案,解析内容,选项A,选项B,选项C,选项D,选项E,选项F,选项G

作者：AI Assistant
"""

import csv
import os
import sys
from pathlib import Path
from datetime import datetime

def convert_answer_format(answer, question_type):
    """
    转换答案格式
    单选题：A/B/C/D保持不变
    多选题：ABC等保持不变
    判断题：需要特殊处理（当前数据中没有）
    """
    if question_type == "判断":
        # 判断题的答案转换逻辑（如果有的话）
        if answer.lower() in ["对", "正确", "是", "true", "t"]:
            return "正确"
        elif answer.lower() in ["错", "错误", "否", "false", "f"]:
            return "错误"
        else:
            return answer
    else:
        # 单选题和多选题保持原格式
        return answer.strip()

def convert_csv(input_file, output_file=None):
    """
    转换CSV文件格式

    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径，如果不指定则自动生成
    """
    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在：{input_file}")
        return False

    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / "template_import.csv")

    try:
        with open(input_file, 'r', encoding='utf-8-sig') as infile, \
             open(output_file, 'w', encoding='utf-8-sig', newline='') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # 跳过表头
            header = next(reader)

            for row in reader:
                if len(row) < 9:  # 确保有足够的列（现在需要至少9列：序号到答案）
                    continue

                # 解析输入数据
                seq_num = row[0]  # 序号
                question_type = row[1]  # 题型
                question = row[2]  # 题干
                option_a = row[3]  # 选项A
                option_b = row[4]  # 选项B
                option_c = row[5]  # 选项C
                option_d = row[6]  # 选项D
                option_e = row[7] if len(row) > 7 else ""  # 选项E
                answer = row[8] if len(row) > 8 else ""  # 答案
                explanation = row[9] if len(row) > 9 else ""  # 解析内容

                # 转换答案格式
                converted_answer = convert_answer_format(answer, question_type)

                # 构建输出行
                output_row = [
                    question,           # 题干
                    converted_answer,    # 答案
                    explanation if explanation.strip() else "解析内容，没有留空",  # 解析内容
                    option_a,           # 选项A
                    option_b,           # 选项B
                    option_c,           # 选项C
                    option_d,           # 选项D
                    option_e,           # 选项E
                    "",                 # 选项F（留空）
                    ""                  # 选项G（留空）
                ]

                writer.writerow(output_row)

        print(f"转换完成！")
        print(f"输入文件：{input_file}")
        print(f"输出文件：{output_file}")
        return True

    except Exception as e:
        print(f"转换过程中出错：{e}")
        return False

def find_latest_output_csv():
    """
    查找最新的output CSV文件
    """
    output_dir = Path("output")
    if not output_dir.exists():
        return None

    # 查找所有CSV文件
    csv_files = []
    for csv_file in output_dir.rglob("*.csv"):
        if csv_file.name == "ocr_results.csv":
            csv_files.append(csv_file)

    if not csv_files:
        return None

    # 按修改时间排序，返回最新的
    return max(csv_files, key=lambda f: f.stat().st_mtime)

def list_available_folders():
    """
    列出所有可用的output文件夹，并显示创建日期
    返回：[(任务文件夹路径, 创建时间), ...] 的列表
    """
    output_dir = Path("output")
    if not output_dir.exists():
        print("错误：output目录不存在")
        return []

    folders = []
    # 查找所有包含ocr_results.csv的文件夹（通常是output/日期/任务ID/的结构）
    for csv_file in output_dir.rglob("ocr_results.csv"):
        task_folder = csv_file.parent  # 任务ID文件夹（如0528d0fd）
        date_folder = task_folder.parent  # 日期文件夹（如20260113_140243）

        # 获取日期文件夹的创建时间
        try:
            # 使用修改时间作为创建时间
            create_time = datetime.fromtimestamp(date_folder.stat().st_mtime)
        except:
            create_time = datetime.now()

        # 显示格式：日期_任务ID
        folder_display_name = f"{date_folder.name}/{task_folder.name}"
        folders.append((csv_file.parent, create_time, folder_display_name))

    # 按创建时间倒序排序（最新的在前面）
    folders.sort(key=lambda x: x[1], reverse=True)
    return folders

def interactive_folder_selection():
    """
    交互式选择文件夹
    返回：选中的文件夹路径，如果用户取消则返回None
    """
    folders = list_available_folders()
    if not folders:
        print("未找到包含ocr_results.csv的文件夹")
        return None

    print("\n📁 可用的任务文件夹列表：")
    print("-" * 70)
    for i, (folder, create_time, display_name) in enumerate(folders, 1):
        time_str = create_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{i:2d}. {display_name} (创建时间: {time_str})")
    print("-" * 70)

    while True:
        try:
            choice = input("请选择任务编号 (1-{}), 或按回车使用最新任务: ".format(len(folders))).strip()

            if not choice:  # 按回车使用最新文件夹
                selected_folder, _, display_name = folders[0]
                print(f"✓ 已选择最新任务：{display_name}")
                return selected_folder

            choice_num = int(choice)
            if 1 <= choice_num <= len(folders):
                selected_folder, _, display_name = folders[choice_num - 1]
                print(f"✓ 已选择任务：{display_name}")
                return selected_folder
            else:
                print(f"❌ 无效选择，请输入1-{len(folders)}之间的数字")
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n❌ 操作已取消")
            return None

def main():
    if len(sys.argv) > 2:
        print("用法：python csv_converter.py [--auto] [输入文件路径]")
        print("  --auto: 自动使用最新文件（不显示交互式选择）")
        return
    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg == "--auto" or arg == "-a":
            # 自动模式：直接使用最新的文件
            input_file = find_latest_output_csv()
            if input_file is None:
                print("未找到output目录中的CSV文件")
                return
        else:
            # 指定文件模式
            input_file = Path(arg)
            if not input_file.exists():
                print(f"错误：指定的输入文件不存在：{input_file}")
                return
    else:
        # 默认交互式模式
        selected_folder = interactive_folder_selection()
        if selected_folder is None:
            return
        input_file = selected_folder / "ocr_results.csv"

    print(f"使用输入文件：{input_file}")
    convert_csv(str(input_file))

if __name__ == "__main__":
    main()
