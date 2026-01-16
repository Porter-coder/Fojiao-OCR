# get_region.py
import pyautogui
import time

print("=" * 50)
print("区域坐标获取工具")
print("=" * 50)
print()
print("步骤：")
print("1. 先把鼠标移到小程序窗口【左上角】，按回车记录")
print("2. 再把鼠标移到小程序窗口【右下角】，按回车记录")
print()

input("准备好后按回车开始...")

print("\n请把鼠标移到小程序窗口【左上角】...")
time.sleep(0.5)

# 实时显示位置
print("当前位置实时显示中，位置OK后按 Ctrl+C")
try:
    while True:
        x, y = pyautogui.position()
        print(f"左上角位置: x={x}, y={y}    ", end='\r')
        time.sleep(0.1)
except KeyboardInterrupt:
    left, top = pyautogui.position()
    print(f"\n✓ 左上角已记录: ({left}, {top})")

print("\n请把鼠标移到小程序窗口【右下角】...")
time.sleep(0.5)

try:
    while True:
        x, y = pyautogui.position()
        print(f"右下角位置: x={x}, y={y}    ", end='\r')
        time.sleep(0.1)
except KeyboardInterrupt:
    right, bottom = pyautogui.position()
    print(f"\n✓ 右下角已记录: ({right}, {bottom})")

width = right - left
height = bottom - top

print("\n" + "=" * 50)
print("截图区域配置（复制到截图脚本中）：")
print("=" * 50)
print(f"LEFT = {left}")
print(f"TOP = {top}")
print(f"WIDTH = {width}")
print(f"HEIGHT = {height}")
print()
print(f"# 或者一行：")
print(f"REGION = ({left}, {top}, {width}, {height})")
print("=" * 50)
