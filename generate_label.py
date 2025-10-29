import os

# 设置目标文件夹和输出文件名
target_dir = "processed_data2"
output_file = "label.txt"

# 收集所有 .h5 文件的文件名（不带路径）
h5_filenames = []
for root, _, files in os.walk(target_dir):
    for file in files:
        if file.endswith(".h5"):
            h5_filenames.append(file)

# 写入到 label.txt
with open(output_file, "w") as f:
    for filename in sorted(h5_filenames):
        f.write(f"{filename}\n")

print(f"✅ 共写入 {len(h5_filenames)} 个 .h5 文件到 {output_file}")
