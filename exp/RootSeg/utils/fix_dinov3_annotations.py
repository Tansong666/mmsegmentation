"""
批量为 DINOv3 文件添加 'from __future__ import annotations' 以解决 Python 3.9 兼容性问题
"""
import os
import re

# 需要修复的文件列表
files_to_fix = [
    "dinov3/layers/patch_embed.py",
    "dinov3/layers/block.py",
    "dinov3/layers/attention.py",
    "dinov3/eval/segmentation/config.py",
    "dinov3/eval/segmentation/schedulers.py",
    "dinov3/eval/segmentation/inference.py",
    "dinov3/eval/setup.py",
    "dinov3/hub/backbones.py",
]

base_dir = "e:/Project/mmsegmentation/exp/RootSeg"

for rel_path in files_to_fix:
    file_path = os.path.join(base_dir, rel_path)
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经有 from __future__ import annotations
    if 'from __future__ import annotations' in content:
        print(f"Already fixed: {rel_path}")
        continue
    
    # 找到版权声明块的结尾（连续的 # 注释行）
    lines = content.split('\n')
    insert_pos = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#') or stripped == '':
            insert_pos = i + 1
        else:
            break
    
    # 在注释块后插入 future import
    lines.insert(insert_pos, '')
    lines.insert(insert_pos + 1, 'from __future__ import annotations')
    
    new_content = '\n'.join(lines)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Fixed: {rel_path}")

print("\nDone!")
