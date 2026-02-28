"""
一键部署脚本 — 将训练输出复制到 AI 推理固件工程
==============================================
自动将 model-training/output/ 下生成的 C 代码
复制到 firmware/ai-inference/applications/ 中。

使用方法：
    python deploy_model.py
"""

import os
import shutil

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

SOURCE_DIR = os.path.join(PROJECT_ROOT, 'model-training', 'output')
TARGET_DIR = os.path.join(PROJECT_ROOT, 'firmware', 'ai-inference', 'applications')

# 需要复制的文件
FILES_TO_DEPLOY = [
    'nn_model.c',
    'nn_model.h',
    'preprocessing.c',
    'preprocessing.h',
]

def main():
    print("=" * 50)
    print("  Edgi-Talk AI 模型部署工具")
    print("=" * 50)
    print(f"  源目录: {SOURCE_DIR}")
    print(f"  目标目录: {TARGET_DIR}")
    print()

    if not os.path.exists(SOURCE_DIR):
        print(f"  ❌ 源目录不存在: {SOURCE_DIR}")
        print(f"     请先运行模型训练: cd model-training && python run_training.py")
        return False

    if not os.path.exists(TARGET_DIR):
        print(f"  ❌ 目标目录不存在: {TARGET_DIR}")
        print(f"     请确保 firmware/ai-inference 工程结构完整")
        return False

    success = 0
    failed = 0

    for fname in FILES_TO_DEPLOY:
        src = os.path.join(SOURCE_DIR, fname)
        dst = os.path.join(TARGET_DIR, fname)

        if not os.path.exists(src):
            print(f"  ⚠️  跳过 {fname} (源文件不存在)")
            failed += 1
            continue

        # 备份已有文件
        if os.path.exists(dst):
            backup = dst + '.bak'
            shutil.copy2(dst, backup)
            print(f"  📦 备份: {fname} → {fname}.bak")

        shutil.copy2(src, dst)
        src_size = os.path.getsize(src)
        print(f"  ✅ 部署: {fname} ({src_size:,} bytes)")
        success += 1

    print()
    print(f"  部署完成: {success} 成功, {failed} 跳过")

    if success > 0:
        print()
        print("  下一步:")
        print("  1. 使用 RT-Thread Studio 打开 firmware/ai-inference/")
        print("  2. 编译并烧录至开发板")
        print("  3. 开发板上电后即可进行实时动作识别")

    return failed == 0

if __name__ == '__main__':
    main()
