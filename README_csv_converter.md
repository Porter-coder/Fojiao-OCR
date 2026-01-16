# 佛脚刷题OCR处理器

专门为微信小程序"佛脚刷题"设计的智能OCR识别工具，支持自动截图、题目提取和格式转换。

## 🎯 项目用途

- 📱 **小程序专用**：专为微信小程序"佛脚刷题"优化
- 📸 **自动截图**：智能识别小程序界面并自动截图
- 🔍 **题目提取**：OCR识别题目内容、选项、答案
- 🤖 **AI优化**：DeepSeek AI辅助解析和纠错
- 📋 **格式转换**：转换为标准题库导入格式

## ✨ 主要特性

- 🎯 **精准定位**：自动检测微信小程序"佛脚刷题"位置
- 📸 **智能截图**：定时自动截取题目区域
- 🔍 **高精度OCR**：使用RapidOCR进行文字识别
- 🤖 **AI解析**：集成DeepSeek API智能解析题目
- 📊 **多种输出**：JSON、CSV、TXT格式支持
- ⚡ **GPU加速**：CUDA加速的OCR处理
- 🔄 **断点续传**：支持中断恢复，避免重复处理
- 🛡️ **安全配置**：环境变量管理敏感信息

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/yourusername/question-ocr-processor.git
cd question-ocr-processor

# 安装依赖
pip install -r requirements-gpu.txt
```

### 2. 配置DeepSeek API

**⚠️ 重要**：`.env` 文件不会自动创建，你需要手动配置。

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，填入你的API密钥
# 从 https://platform.deepseek.com/ 获取API密钥
DEEPSEEK_API_KEY=sk-your-actual-api-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPMODEL=deepseek-chat
```

### 3. 使用流程

#### 准备工作
1. 打开微信小程序"佛脚刷题"
2. 将小程序窗口调整到**右上角**位置（默认配置）
3. 确保小程序界面清晰可见

#### 运行流程
```bash
# 1. 自动截图（默认适配微信小程序位置）
python screenshot.py

# 2. OCR识别和AI解析
python ocr_recognize.py

# 3. 转换为题库导入格式
python csv_converter.py
```

#### 截图配置说明
- **默认位置**：微信小程序"佛脚刷题"位于屏幕右上角
- **截图区域**：自动检测小程序窗口并截取题目区域
- **定时截图**：支持连续截图多个题目

## 📋 题库格式转换

### 功能说明

将从"佛脚刷题"小程序OCR识别的结果转换为标准题库导入格式。

### 核心功能

- 🔍 **自动查找**：自动检测最新的OCR识别结果
- 📝 **格式转换**：支持单选题、多选题、判断题
- 🤖 **AI增强**：DeepSeek AI提供题目解析和答案
- 📊 **标准输出**：生成符合题库系统的导入格式

### 输入输出格式

**输入格式**（OCR识别结果）：
```
序号,题型,题干,选项A,选项B,选项C,选项D,选项E,答案
```

**输出格式**（题库导入格式）：
```
题干,答案,解析内容,选项A,选项B,选项C,选项D,选项E,选项F,选项G
```

### 使用方法

#### 自动转换（推荐）
```bash
python csv_converter.py
```
自动查找并转换最新的OCR结果文件。

#### 手动指定文件
```bash
python csv_converter.py path/to/your/ocr_results.csv
```

#### 输出位置
转换后的文件保存在：`output/[日期]/[任务ID]/template_import.csv`

## 💡 使用技巧

### 1. 题目类型识别
- **单选题**：答案格式 A/B/C/D
- **多选题**：答案格式 ABC/ABD 等
- **判断题**：正确/错误

### 2. AI解析模式
- **完整解析**：包含答案、解析、修正说明
- **结构解析**：仅提取题目结构和选项

### 3. 缓存机制
- 避免重复解析相同题目
- 支持不同的解析模式独立缓存

## 📁 项目文件结构

```
FOJIAO/
├── ocr_recognize.py      # OCR识别和AI解析
├── csv_converter.py       # 格式转换工具
├── screenshot.py          # 自动截图脚本
├── requirements-gpu.txt   # GPU环境依赖
├── .env.example          # 环境配置模板
├── output/               # 输出结果目录
├── processing/           # 处理中文件目录
├── screenshot_temp/      # 截图临时文件
└── venv/                 # 虚拟环境（已忽略）
```

## 🔧 系统要求

- **Python**: 3.8+
- **CUDA**: 11.8+ (GPU加速)
- **内存**: 8GB+ 推荐
- **操作系统**: Windows/Linux/macOS

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件
