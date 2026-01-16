# ocr_recognize.py — 用 EasyOCR（GPU加速版本）
import os
import json
import uuid
import shutil
import csv
import re
import logging
import platform
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # 如果没有安装 python-dotenv，继续执行

# 禁用PyTorch RNN警告
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.rnn')

# -------------------- DeepSeek API配置 --------------------
try:
    from openai import OpenAI, APITimeoutError, APIError
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-fbab52c876d64fa2b9a22fd47b4aa6d1")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    DEEPMODEL = os.getenv("DEEPMODEL", "deepseek-chat")

    deepseek_client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        timeout=30,
        max_retries=1
    )
    print("✓ DeepSeek API初始化成功")
    deepseek_available = True
except ImportError:
    print("⚠ 未安装 openai，AI解析功能将不可用")
    deepseek_available = False
except Exception as e:
    print(f"⚠ DeepSeek API初始化失败: {e}")
    deepseek_available = False

# -------------------- AI解析函数 --------------------
def clean_json_output(text: str) -> str:
    """清理AI返回的JSON字符串"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text

def build_ai_parse_prompt(ocr_text: str, question_num: int) -> list:
    """构建AI解析提示词 - 仿照deepseek_process.py"""
    system = """你是一个专业的题目校对与答题助手。学科: 通用题库。
请严格按要求处理题目并返回 JSON。"""

    user = f"""请对以下 OCR 识别的题目进行处理：

1. 纠正题干和选项中的错别字、乱码、病句
2. 分析题目类型（单选/多选/判断）
3. 给出正确答案和详细解析
4. 提供修复理由和判断理由
5. 严格返回 JSON，不要其他文字

原始 OCR 内容：
{ocr_text}

返回 JSON 格式：
{{
    "题号": {question_num},
    "类型": "单选/多选/判断",
    "原始题目": "OCR原文",
    "题目": "纠正后的题目",
    "选项": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "题目已修正": true/false,
    "修正说明": "修正了什么内容（没修正则为空）",
    "答案": "A/B/C/D 或 对/错",(如果是多选，返回AB，ABC，BCD类似格式，不要加分隔符)
    "解析": "答案解析",
    "修复理由": "为什么需要修复OCR识别结果",
    "判断理由": "为什么这样判断题型和答案"
}}

注意：
- 判断题不需要选项字段
- 修复理由：详细说明OCR识别的错误和修正依据
- 判断理由：说明题型判断和答案推断的逻辑"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

def parse_question_with_ai(ocr_text: str, question_num: int) -> dict:
    """使用AI解析题目结构 - 仿照deepseek_process.py"""
    if not deepseek_available:
        raise RuntimeError("DeepSeek API不可用，请安装openai库")

    messages = build_ai_parse_prompt(ocr_text, question_num)

    try:
        response = deepseek_client.chat.completions.create(
            model=DEEPMODEL,
            messages=messages,
            temperature=0
        )

        raw = response.choices[0].message.content
        if not raw:
            raise RuntimeError("AI返回内容为空")

        cleaned = clean_json_output(raw)
        parsed = json.loads(cleaned)

        # 验证必需字段 - 仿照deepseek_process.py
        required_fields = ["题号", "类型", "题目", "选项", "答案", "解析"]
        for field in required_fields:
            if field not in parsed:
                raise RuntimeError(f"AI返回缺少必需字段: {field}")

        # 转换选项格式 - 从{"A": "...", "B": "..."}转换为{"A": "...", "B": "...", ...}
        options = parsed.get("选项", {})
        if isinstance(options, dict):
            parsed["选项A"] = options.get("A", "")
            parsed["选项B"] = options.get("B", "")
            parsed["选项C"] = options.get("C", "")
            parsed["选项D"] = options.get("D", "")
            parsed["选项E"] = options.get("E", "")

        return parsed

    except APITimeoutError:
        raise RuntimeError("AI解析超时")
    except APIError as e:
        raise RuntimeError(f"AI API错误: {str(e)}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"AI返回JSON格式错误: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"AI解析失败: {str(e)}")

# GPU加速相关环境变量设置 - 在导入前设置
os.environ['OMP_NUM_THREADS'] = '1'  # 避免线程冲突
os.environ['MKL_NUM_THREADS'] = '1'

try:
    import easyocr
    print("✓ 使用EasyOCR (支持GPU加速)")
except ImportError:
    print("错误：未安装 easyocr")
    print("请运行: pip install easyocr")
    exit(1)

# 创建任务ID和目录结构
import time
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")  # 时间戳到秒
TASK_ID = str(uuid.uuid4())[:8]  # 使用8位UUID作为任务ID
SCREENSHOT_DIR = "screenshots"
SCREENSHOT_TEMP_DIR = f"screenshot_temp/{TIMESTAMP}_{TASK_ID}"
PROCESSING_DIR = f"processing/{TIMESTAMP}/{TASK_ID}/ocr"  # 中间结果目录
OUTPUT_DIR = f"output/{TIMESTAMP}/{TASK_ID}"  # 最终结果目录
OCR_THREADS = 12

# 创建任务专用的目录结构
os.makedirs(SCREENSHOT_TEMP_DIR, exist_ok=True)
os.makedirs(f"{PROCESSING_DIR}/images", exist_ok=True)  # 中间结果：标注图片
os.makedirs(f"{PROCESSING_DIR}/texts", exist_ok=True)   # 中间结果：文本文件
os.makedirs(f"{PROCESSING_DIR}/details", exist_ok=True) # 中间结果：详细JSON
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 最终结果目录

# 移动截图文件到临时目录
if os.path.exists(SCREENSHOT_DIR) and os.listdir(SCREENSHOT_DIR):
    print(f"移动截图文件到临时目录: {SCREENSHOT_TEMP_DIR}")
    for filename in os.listdir(SCREENSHOT_DIR):
        src_path = os.path.join(SCREENSHOT_DIR, filename)
        dst_path = os.path.join(SCREENSHOT_TEMP_DIR, filename)
        if os.path.isfile(src_path):
            shutil.move(src_path, dst_path)
    print(f"✓ 已移动 {len(os.listdir(SCREENSHOT_TEMP_DIR))} 个文件")
else:
    print(f"⚠ 源目录 {SCREENSHOT_DIR} 不存在或为空")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ocr")

def flatten_rnn_parameters(model):
    """递归地压缩RNN模型的权重到连续内存块"""
    try:
        import torch
        for module in model.modules():
            if isinstance(module, (torch.nn.RNNBase, torch.nn.LSTM, torch.nn.GRU)):
                try:
                    module.flatten_parameters()
                except RuntimeError:
                    # 如果已经在连续内存中，flatten_parameters()会抛出异常，忽略即可
                    pass
    except ImportError:
        # 如果没有torch，跳过这个优化
        pass

def init_ocr_with_gpu():
    """初始化EasyOCR"""
    try:
        print("正在初始化 EasyOCR...")

        # 检查GPU可用性
        gpu_available = check_gpu_availability()

        if gpu_available:
            print("✓ 检测到GPU，EasyOCR将使用GPU加速")
            # EasyOCR自动检测GPU
            reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
        else:
            print("⚠ 未检测到GPU，EasyOCR将使用CPU模式")
            reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

        # 修复RNN权重内存连续性警告
        try:
            import torch
            if hasattr(reader, 'recognition_network'):
                flatten_rnn_parameters(reader.recognition_network)
                print("✓ RNN权重已压缩到连续内存")
            elif hasattr(reader, 'recognizer'):
                flatten_rnn_parameters(reader.recognizer)
                print("✓ RNN权重已压缩到连续内存")
        except Exception as e:
            # 如果无法访问内部模型，忽略这个优化
            pass

        print("✓ EasyOCR加载完成！")
        return reader, gpu_available

    except Exception as e:
        logger.error(f"EasyOCR初始化失败: {e}")
        raise e

def check_gpu_availability():
    """检查GPU是否可用"""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ PyTorch检测到GPU: {gpu_name} ({gpu_count}个)")
            return True
        else:
            print("⚠ PyTorch未检测到GPU")
            return False
    except ImportError:
        print("⚠ 未安装PyTorch，使用CPU模式")
        return False
    except Exception as e:
        print(f"GPU检测失败: {e}")
        return False

def optimize_thread_count(use_gpu):
    """根据是否使用GPU优化线程数"""
    if use_gpu:
        # GPU加速时减少线程数，避免GPU内存不足
        return min(4, os.cpu_count() or 4)
    else:
        # CPU模式使用更多线程
        return min(12, os.cpu_count() or 8)

# 检查GPU可用性
gpu_available = check_gpu_availability()

print(f"任务ID: {TASK_ID}")
print("正在初始化 RapidOCR...")
ocr, use_gpu = init_ocr_with_gpu()
print(f"使用模式: {'GPU加速' if use_gpu else 'CPU'}")

# 根据GPU使用情况优化线程数
OCR_THREADS = optimize_thread_count(use_gpu)
print(f"优化线程数: {OCR_THREADS}\n")

def parse_question_type(lines):
    for line in lines:
        if "单选" in line: return "单选"
        if "多选" in line: return "多选"
        if "判断" in line: return "判断"
    return "未知"

def parse_options(lines):
    """改进的选项解析，支持跨行选项"""
    opt_map = {"A": "", "B": "", "C": "", "D": "", "E": ""}

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # 检查是否是选项开始行 (A. B. C. D. E.)
        match = re.match(r'^([A-E])[.．、:\s]*(.*)', line)
        if match:
            option_key = match.group(1)
            option_content = match.group(2).strip()

            # 如果选项内容为空，尝试合并下一行
            if not option_content and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # 如果下一行不以选项字母开头，就合并
                if next_line and not re.match(r'^[A-E][.．、:\s]', next_line):
                    option_content = next_line
                    i += 1  # 跳过下一行

            opt_map[option_key] = f"{option_key}. {option_content}" if option_content else f"{option_key}."
        i += 1

    return opt_map

def save_results_to_csv(results, output_path):
    """将OCR结果保存为CSV格式 - 序号+题型+题干+选项A+选项B+选项C+选项D+选项E+答案+解析"""
    if not results:
        return

    # CSV头部 - 仿照deepseek_process.py
    headers = ['序号', '题型', '题干', '选项A', '选项B', '选项C', '选项D', '选项E', '答案', '解析', '已修正', '修正说明']

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for result in results:
            # 直接使用AI解析的结果
            options = result.get('选项', {})

            row = {
                '序号': result.get('序号', ''),
                '题型': result.get('类型', ''),
                '题干': result.get('题干', ''),
                '选项A': options.get('A', ''),
                '选项B': options.get('B', ''),
                '选项C': options.get('C', ''),
                '选项D': options.get('D', ''),
                '选项E': options.get('E', ''),
                '答案': result.get('答案', ''),
                '解析': result.get('解析', '').replace('\n', ' '),
                '已修正': '是' if result.get('题目已修正') else '否',
                '修正说明': result.get('修正说明', '')
            }
            writer.writerow(row)

def filter_answer_context(details):
    """
    完全过滤掉答案相关的内容及其上下文
    包括：我的答案、参考答案、答案解析及其后面的所有相关内容
    """
    filtered = []
    skip_mode = False  # 是否处于跳过模式
    skip_keywords = ["我的答案", "参考答案", "答案解析"]

    for i, line in enumerate(details):
        text = line["text"].strip()

        # 检查是否遇到需要跳过的关键词
        found_skip_keyword = False
        for keyword in skip_keywords:
            if keyword in text:
                found_skip_keyword = True
                skip_mode = True
                break

        # 如果找到跳过关键词，开始跳过模式
        if found_skip_keyword:
            continue

        # 如果处于跳过模式，继续跳过
        if skip_mode:
            continue

        # 正常内容，添加到过滤结果中
        line_copy = line.copy()
        line_copy["index"] = len(filtered) + 1
        filtered.append(line_copy)

    return filtered

def save_annotated_image(image_path, boxes, output_path):
    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("msyh.ttc", 14)
        except:
            font = ImageFont.load_default()
        
        for i, item in enumerate(boxes):
            box = item['poly']
            conf = item['confidence']
            points = [(int(p[0]), int(p[1])) for p in box]
            points.append(points[0])
            draw.line(points, fill="red", width=2)
            draw.text((int(box[0][0]), int(box[0][1])-18), f"{i+1}|{conf:.2f}", fill="blue", font=font)
        
        img.save(output_path)
    except Exception as e:
        logger.warning(f"保存标注图片失败: {e}")

def process_image(item):
    idx, filename = item
    image_path = os.path.join(SCREENSHOT_TEMP_DIR, filename)
    base_name = os.path.splitext(filename)[0]

    try:
        # EasyOCR API
        import time
        start_time = time.time()

        result = ocr.readtext(image_path)
        elapse = time.time() - start_time
    except Exception as e:
        logger.error(f"OCR 失败 {filename}: {e}")
        return {"序号": idx + 1, "文件": filename, "错误": str(e)}
    
    if not result:
        return {"序号": idx + 1, "文件": filename, "错误": "OCR 返回为空"}
    
    # 处理EasyOCR结果
    raw_details = []
    try:
        # EasyOCR返回格式: [[bbox, text, confidence], ...]
        for i, (bbox, text, confidence) in enumerate(result):
            raw_details.append({
                "index": i + 1,
                "text": text,
                "confidence": round(float(confidence), 4),
                "poly": [[int(p[0]), int(p[1])] for p in bbox]
            })

    except Exception as e:
        print(f"结果解析错误: {e}")
        print(f"结果类型: {type(result)}")
        if result:
            print(f"第一个结果: {result[0] if len(result) > 0 else '空'}")

    # 过滤答案相关内容
    details = filter_answer_context(raw_details)

    # 从过滤后的details生成texts
    texts = [line["text"] for line in details]

    full_text = "\n".join(texts)
    lines = [t.strip() for t in texts if t.strip()]
    
    # 保存中间结果到processing目录
    with open(f"{PROCESSING_DIR}/texts/{base_name}.txt", "w", encoding="utf-8") as f:
        for i, t in enumerate(texts):
            f.write(f"[{i+1}] {t}\n")

    with open(f"{PROCESSING_DIR}/details/{base_name}.json", "w", encoding="utf-8") as f:
        json.dump({"file": filename, "total_lines": len(texts), "lines": details}, f, ensure_ascii=False, indent=2)

    save_annotated_image(image_path, details, f"{PROCESSING_DIR}/images/{base_name}_ocr.png")

    # 只返回OCR结果，后续统一进行AI解析
    result = {
        "序号": idx + 1,
        "文件": filename,
        "原始文本": full_text,
        "行数": len(texts),
        "解析状态": "待AI解析"
    }

    # 调试信息
    if idx < 3:  # 只打印前3个结果的调试信息
        print(f"OCR完成 {filename}: {len(texts)} 行文本")
    return result

def cleanup_empty_temp_dirs():
    """清理空的temp目录"""
    if not os.path.exists("screenshot_temp"):
        return

    cleaned_count = 0
    for dir_name in os.listdir("screenshot_temp"):
        temp_path = os.path.join("screenshot_temp", dir_name)
        if os.path.isdir(temp_path):
            # 检查目录是否为空或只包含非图片文件
            try:
                files = os.listdir(temp_path)
                image_count = len([f for f in files
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if image_count == 0:
                    # 删除空目录
                    import shutil
                    shutil.rmtree(temp_path)
                    cleaned_count += 1
                    print(f"已清理空temp目录: {dir_name}")
            except Exception as e:
                print(f"清理目录 {dir_name} 时出错: {e}")

    if cleaned_count > 0:
        print(f"共清理了 {cleaned_count} 个空temp目录")
    return cleaned_count

def select_data_source():
    """交互式选择数据源"""
    print("=== 数据源选择 ===")

    # 首先清理空的temp目录
    cleanup_empty_temp_dirs()
    print()

    # 检查现有的screenshot_temp目录
    temp_dirs = []
    if os.path.exists("screenshot_temp"):
        temp_dirs = [d for d in os.listdir("screenshot_temp")
                    if os.path.isdir(os.path.join("screenshot_temp", d))]

    # 检查screenshots目录
    has_new_screenshots = os.path.exists(SCREENSHOT_DIR) and \
                         any(f.lower().endswith(('.png', '.jpg', '.jpeg'))
                             for f in os.listdir(SCREENSHOT_DIR))

    print("可用数据源:")
    print("0. 重新截取新的截图")

    for i, temp_dir in enumerate(temp_dirs, 1):
        temp_path = os.path.join("screenshot_temp", temp_dir)
        png_count = len([f for f in os.listdir(temp_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"{i}. 使用现有截图: {temp_dir} ({png_count}张图片)")

    if has_new_screenshots:
        new_count = len([f for f in os.listdir(SCREENSHOT_DIR)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"{len(temp_dirs)+1}. 使用screenshots目录 ({new_count}张图片)")

    while True:
        try:
            choice = input("\n请选择数据源 (输入数字): ").strip()

            if choice == "0":
                print("请先截取新的截图到screenshots目录，然后重新运行程序")
                return None, None

            choice_num = int(choice)

            if 1 <= choice_num <= len(temp_dirs):
                selected_dir = os.path.join("screenshot_temp", temp_dirs[choice_num-1])
                print(f"✓ 选择使用: {temp_dirs[choice_num-1]}")
                return selected_dir, temp_dirs[choice_num-1]

            elif has_new_screenshots and choice_num == len(temp_dirs) + 1:
                selected_dir = SCREENSHOT_DIR
                print("✓ 选择使用screenshots目录")
                return selected_dir, "new_screenshots"

            else:
                print("无效选择，请重新输入")

        except ValueError:
            print("请输入有效数字")

def select_parsing_mode():
    """
    交互式选择AI解析模式
    返回：(parsing_mode, mode_name)
    parsing_mode: "full" (完整答案) 或 "structure" (仅题库结构)
    """
    print("\n" + "="*50)
    print("🤖 选择AI解析模式")
    print("="*50)
    print("1. 完整答案模式：AI解析出完整的答案和详细解析")
    print("2. 题库结构模式：AI只解析题库结构，不解析答案")
    print("="*50)

    while True:
        try:
            choice = input("请选择模式 (1-2), 或按回车使用完整答案模式: ").strip()

            if not choice:  # 按回车默认完整答案模式
                print("✓ 已选择：完整答案模式")
                return "full", "完整答案模式"

            choice_num = int(choice)
            if choice_num == 1:
                print("✓ 已选择：完整答案模式")
                return "full", "完整答案模式"
            elif choice_num == 2:
                print("✓ 已选择：题库结构模式")
                return "structure", "题库结构模式"
            else:
                print("❌ 无效选择，请输入1或2")
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n❌ 操作已取消")
            return None, None

def main():
    import time
    start_time = time.time()

    # 交互式选择数据源
    selected_source, source_name = select_data_source()
    if selected_source is None:
        return

    # 交互式选择AI解析模式
    parsing_mode, mode_name = select_parsing_mode()
    if parsing_mode is None:
        return

    # 设置数据源
    global SCREENSHOT_TEMP_DIR
    if source_name != "new_screenshots":
        # 使用现有temp目录
        SCREENSHOT_TEMP_DIR = selected_source
    # 如果选择new_screenshots，保持原有逻辑（会移动文件）

    if not os.path.isdir(SCREENSHOT_TEMP_DIR):
        print(f"错误：找不到 {SCREENSHOT_TEMP_DIR} 文件夹")
        return

    files = sorted([f for f in os.listdir(SCREENSHOT_TEMP_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not files:
        print(f"{SCREENSHOT_TEMP_DIR} 中没有图片！")
        return

    print(f"共 {len(files)} 张图片，开始 OCR 识别...")
    print(f"任务ID: {TASK_ID}")
    print(f"临时截图目录: {SCREENSHOT_TEMP_DIR}")
    print(f"中间结果目录: processing/{TIMESTAMP}/{TASK_ID}/ocr/")
    print(f"最终结果目录: output/{TIMESTAMP}/{TASK_ID}/")
    print(f"系统信息: {platform.system()} {platform.release()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"GPU加速: {'启用' if use_gpu else '禁用'}")
    print("-" * 60)
    
    results = []
    errors = []
    
    items = list(enumerate(files))
    with ThreadPoolExecutor(max_workers=OCR_THREADS) as executor:
        futures = {executor.submit(process_image, it): it for it in items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="OCR"):
            try:
                res = fut.result()
                if res is None:
                    print("警告：process_image返回None")
                    continue
                if "错误" in res:
                    errors.append(res)
                    print(f"错误结果: {res.get('文件', 'unknown')} - {res.get('错误', 'unknown')}")
                else:
                    results.append(res)
                    if len(results) <= 3:  # 只打印前3个成功结果
                        print(f"成功结果: {res.get('文件', 'unknown')} - {res.get('行数', 0)} 行")
            except Exception as e:
                logger.exception(f"处理异常: {e}")
                print(f"异常详情: {e}")
    
    results.sort(key=lambda x: x.get("序号", 0))

    # ========== AI并发解析 ==========
    if deepseek_available and results and parsing_mode:
        print(f"\n开始AI并发解析 {len(results)} 道题目...")

        # 检查缓存文件
        cache_file = f"{PROCESSING_DIR}/ai_cache.json"
        ai_cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    ai_cache = json.load(f)
                print(f"加载AI缓存: {len(ai_cache)} 个已解析题目")
            except Exception as e:
                print(f"加载缓存失败: {e}")

        def ai_parse_task(question_data):
            """AI解析单个题目 - 根据模式决定解析内容"""
            seq_num = question_data["序号"]

            # 检查缓存
            cache_key = f"q_{seq_num}_{parsing_mode}"  # 不同模式使用不同缓存
            if cache_key in ai_cache:
                print(f"使用缓存: 题目 {seq_num} ({parsing_mode}模式)")
                cached_data = ai_cache[cache_key]
                question_data.update(cached_data)
                return question_data

            try:
                parsed = parse_question_with_ai(question_data["原始文本"], question_data["序号"])

                # 根据解析模式决定要更新的字段
                update_data = {
                    "类型": parsed.get("类型", "未知"),
                    "原始题目": parsed.get("原始题目", ""),
                    "题目": parsed.get("题目", ""),
                    "题干": parsed.get("题目", ""),  # 题干使用纠正后的题目
                    "选项": {
                        "A": parsed.get("选项A", ""),
                        "B": parsed.get("选项B", ""),
                        "C": parsed.get("选项C", ""),
                        "D": parsed.get("选项D", ""),
                        "E": parsed.get("选项E", "")
                    },
                    "题目已修正": parsed.get("题目已修正", False),
                    "修正说明": parsed.get("修正说明", ""),
                    "解析状态": f"AI解析成功 ({mode_name})"
                }

                # 根据模式决定是否包含答案相关信息
                if parsing_mode == "full":
                    # 完整答案模式：包含所有解析内容
                    update_data.update({
                        "答案": parsed.get("答案", ""),
                        "解析": parsed.get("解析", ""),
                        "修复理由": parsed.get("修复理由", ""),
                        "判断理由": parsed.get("判断理由", ""),
                    })
                elif parsing_mode == "structure":
                    # 题库结构模式：只解析结构，不解析答案
                    update_data.update({
                        "答案": "",  # 不解析答案
                        "解析": "",  # 不解析详细解析
                        "修复理由": "",
                        "判断理由": "",
                    })
                question_data.update(update_data)

                # 保存到缓存
                ai_cache[cache_key] = update_data.copy()
                try:
                    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(ai_cache, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"保存缓存失败: {e}")

                return question_data
            except Exception as e:
                # AI解析失败，回退到正则解析
                lines = [line.strip() for line in question_data["原始文本"].strip().split('\n') if line.strip()]
                update_data = {
                    "类型": parse_question_type(lines),
                    "选项": parse_options(lines),
                    "题干": "",  # 正则解析不提取题干
                    "题目已修正": False,
                    "修正说明": "",
                    "答案": "",
                    "解析": "",
                    "修复理由": "",
                    "判断理由": "",
                    "解析状态": f"AI解析失败，回退正则: {str(e)}"
                }
                question_data.update(update_data)

                # 保存到缓存（即使是失败的结果）
                ai_cache[cache_key] = update_data.copy()
                try:
                    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(ai_cache, f, ensure_ascii=False, indent=2)
                except Exception as cache_e:
                    print(f"保存缓存失败: {cache_e}")

                return question_data

        # 并发AI解析
        AI_THREADS = 20  # DeepSeek支持20并发
        with ThreadPoolExecutor(max_workers=AI_THREADS) as executor:
            futures = {executor.submit(ai_parse_task, q): q for q in results}

            with tqdm(total=len(futures), desc="AI解析", initial=0) as pbar:
                for fut in as_completed(futures):
                    try:
                        updated_question = fut.result()
                        # 更新原results中的对应项
                        for i, q in enumerate(results):
                            if q["序号"] == updated_question["序号"]:
                                results[i] = updated_question
                                break
                    except Exception as e:
                        logger.error(f"AI解析异常: {e}")
                    finally:
                        pbar.update(1)

        print(f"AI解析完成！({mode_name})")

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 保存最终结果到output目录
    # 机器可读格式
    with open(f"{OUTPUT_DIR}/ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    save_results_to_csv(results, f"{OUTPUT_DIR}/ocr_results.csv")

    # 保存判断理由单独的JSON
    reasoning_data = []
    for r in results:
        reasoning_data.append({
            "题号": r.get("序号"),
            "判断理由": r.get("判断理由", ""),
            "修复理由": r.get("修复理由", "")
        })

    with open(f"{OUTPUT_DIR}/reasoning.json", "w", encoding="utf-8") as f:
        json.dump(reasoning_data, f, ensure_ascii=False, indent=2)

    # 人类友好的格式
    with open(f"{OUTPUT_DIR}/ocr_summary.txt", "w", encoding="utf-8") as f:
        f.write("OCR识别结果汇总\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"任务ID: {TASK_ID}\n")
        f.write(f"处理时间: {TIMESTAMP}\n")
        f.write(f"总文件数: {len(results)}\n")
        f.write(f"成功识别: {len(results)} 个文件\n\n")

        for r in results:
            f.write(f"[{r['序号']}] {r['文件']} ({r['类型']})\n")
            f.write(f"识别行数: {r['行数']}\n")
            if r.get('选项'):
                options = [f"{k}:{v}" for k, v in r['选项'].items() if v]
                if options:
                    f.write(f"选项: {' | '.join(options)}\n")
            f.write(f"文本内容:\n{r.get('原始文本', '')}\n")
            f.write("-" * 30 + "\n\n")
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_image = total_time / len(files) if files else 0

    print("\n" + "=" * 60)
    print("OCR 处理完成！")
    print(f"成功: {len(results)} | 失败: {len(errors)}")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均每张图片: {avg_time_per_image:.2f}秒")
    print(f"处理速度: {len(results)/total_time:.2f} 张/秒" if total_time > 0 else "处理速度: N/A")
    print(f"中间结果: processing/{TIMESTAMP}/{TASK_ID}/ocr/")
    print(f"最终结果: output/{TIMESTAMP}/{TASK_ID}/")
    print("  - ocr_results.json")
    print("  - ocr_results.csv")
    print("  - ocr_summary.txt")
    if parsing_mode:
        print("  - reasoning.json")
    print("=" * 60)

    if errors:
        print("\n失败列表:")
        for e in errors:
            print(f"  - {e.get('文件', '?')}: {e.get('错误', '?')}")

    # GPU加速性能提示
    if use_gpu:
        print("\n💡 GPU加速已启用，性能数据如上所示")
    else:
        print("\n💡 当前使用CPU模式，如需GPU加速请安装CUDA和相应依赖")
        print("   安装指南: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html")

if __name__ == "__main__":
    main()
