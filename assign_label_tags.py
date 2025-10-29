import os
import glob

label_file = "label.txt"
marker_dir = "data/file_markers_detection"

# 将所有 marker 文件读取合并为 dict
label_dict = {}

# 遍历目录下所有标注 .txt 文件
for file_path in glob.glob(os.path.join(marker_dir, "*.txt")):
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                fname, tag = parts
                label_dict[fname] = tag

# 输出检查
print(f"✅ 总共从标注文件中读取了 {len(label_dict)} 个样本标签")

# 读取 label.txt 并补上标签
new_lines = []
missing = []

with open(label_file, 'r') as f:
    for line in f:
        fname = line.strip()
        if fname in label_dict:
            tag = label_dict[fname]
            new_lines.append(f"{fname},{tag}\n")
        else:
            new_lines.append(f"{fname},?\n")
            missing.append(fname)

# 写回 label.txt（就地覆盖）
with open(label_file, 'w') as f:
    f.writelines(new_lines)

print(f"✅ 已更新 label.txt 文件，共 {len(new_lines)} 行")
if missing:
    print(f"⚠️ 找不到标签的样本数: {len(missing)}")
    for m in missing[:10]:
        print(f"  - {m}")
