# get_position.py
import pyautogui
import time

print("5秒后开始显示鼠标位置")
print("请把鼠标移到'下一题'按钮上，记下坐标")
print("按 Ctrl+C 停止")
time.sleep(1)

try:
    while True:
        x, y = pyautogui.position()
        print(f"鼠标位置: x={x}, y={y}    ", end='\r')
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n已停止")
