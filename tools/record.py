"""
PC 端串口数据录入脚本
=====================
通过 USB CDC 虚拟串口从 Edgi-Talk 开发板读取 IMU 加速度数据，
并保存为 CSV 文件，用于后续模型训练。

使用方法：
    1. 烧录 data-collector 固件到开发板
    2. USB 连接开发板到 PC
    3. 修改下方 COM_PORT 为实际串口号
    4. 运行: python record.py
    5. 在开发板 LCD 上点击 [Start Recording] 开始录制
    6. 按 Ctrl+C 结束录制

依赖：
    pip install pyserial
"""

import serial
import csv
import sys
import os
from datetime import datetime

# === 配置参数 ===
COM_PORT = 'COM11'          # 请修改为开发板连接后的实际 COM 口号
BAUD_RATE = 115200
FILE_NAME = None             # 设为 None 则自动生成带时间戳的文件名

def main():
    # 自动生成文件名
    if FILE_NAME is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'action_{timestamp}.csv'
    else:
        output_file = FILE_NAME

    # 确定输出路径
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, output_file)

    print(f"==========================================")
    print(f"  Edgi-Talk IMU 数据录入工具")
    print(f"==========================================")
    print(f"  串口: {COM_PORT}")
    print(f"  波特率: {BAUD_RATE}")
    print(f"  输出文件: {output_path}")
    print(f"")
    print(f"  请在开发板屏幕上点击 [Start Recording] 开始录制")
    print(f"  按 Ctrl+C 结束录制")
    print(f"==========================================")

    sample_count = 0

    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['AccX', 'AccY', 'AccZ'])  # 表头

            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 3:
                        try:
                            # 验证是数值数据
                            [int(p) for p in parts]
                            writer.writerow(parts)
                            sample_count += 1
                            if sample_count % 50 == 0:
                                print(f"  已录制 {sample_count} 个采样点...")
                            elif sample_count <= 5:
                                print(f"  写入: {line}")
                        except ValueError:
                            pass  # 跳过非数值行

    except KeyboardInterrupt:
        print(f"\n  录制结束！")
        print(f"  共 {sample_count} 个采样点")
        print(f"  已保存至: {output_path}")
    except serial.SerialException as e:
        print(f"\n  串口错误: {e}")
        print(f"  请检查 COM_PORT 设置和 USB 连接")
        sys.exit(1)
    except Exception as e:
        print(f"\n  出错: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
