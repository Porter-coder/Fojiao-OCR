# screenshot.py
import pyautogui
import time
import os

# ============ 配置区域（需要修改）============
TOTAL_QUESTIONS = 23      # 总题数
# 下一题按钮坐标
NEXT_BUTTON_X = 666           
NEXT_BUTTON_Y = 1330            

# 截图区域 (左上角x, 左上角y, 宽度, 高度)
# 设为 None 则全屏截图
REGION = (16, 14, 695, 1279)
# 例如: REGION = (100, 150, 400, 700)

DELAY = 0.1                    # 每题间隔秒数
# ============================================

# 创建目录
os.makedirs("screenshots", exist_ok=True)

print("=" * 50)
print("自动截图脚本")
print("=" * 50)
print(f"总题数: {TOTAL_QUESTIONS}")
print(f"下一题按钮: ({NEXT_BUTTON_X}, {NEXT_BUTTON_Y})")
print(f"截图区域: {REGION if REGION else '全屏'}")
print()
input("准备好后按回车开始...")

print("\n3秒后开始截图...")
time.sleep(3)

for i in range(TOTAL_QUESTIONS):
    # 截图
    if REGION:
        screenshot = pyautogui.screenshot(region=REGION)
    else:
        screenshot = pyautogui.screenshot()
    
    filename = f"screenshots/q_{i+1:04d}.png"
    screenshot.save(filename)
    print(f"[{i+1}/{TOTAL_QUESTIONS}] 已保存: {filename}")
    
    # 点击下一题
    if i < TOTAL_QUESTIONS - 1:
        pyautogui.click(NEXT_BUTTON_X, NEXT_BUTTON_Y)
        time.sleep(DELAY)

print("\n截图完成！")
print(f"共 {TOTAL_QUESTIONS} 张图片保存到 screenshots 文件夹")
