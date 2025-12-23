import os

# ===================== 配置项（请修改！）=====================
# 图片文件夹路径（必填！替换为你的图片文件夹绝对路径）
FOLDER_PATH = r"D:\user\code\AI\其他\成绩脚本\修改图片\全部"  # 示例：r"C:\Users\XXX\Pictures\学生照片"
# 支持的图片格式（可根据需要增删）
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
# 是否开启「测试模式」（仅预览重命名结果，不实际修改）
TEST_MODE = False  # 确认预览结果后，改为False执行真实重命名


# ===================== 核心重命名逻辑 =====================
def extract_student_name(filename_no_ext):
    """从文件名（无后缀）中提取学生名字（第一个-后，第二个-前）"""
    # 按横杠分割
    parts = filename_no_ext.split('-')
    # 检查分割后的部分是否足够（至少3部分：xxx-名字-xxx）
    if len(parts) >= 3:
        student_name = parts[1].strip()  # 提取第二个部分（索引1），并去除空格
        # 过滤空名字（防止分割后是空白）
        if student_name:
            return student_name
    # 格式不符合时返回None
    return None


def batch_rename_images():
    # 检查文件夹是否存在
    if not os.path.exists(FOLDER_PATH):
        print(f"❌ 错误：指定的文件夹路径不存在 → {FOLDER_PATH}")
        return

    # 存储待重命名的文件信息
    rename_list = []
    error_list = []

    # 遍历文件夹中的图片
    for filename in os.listdir(FOLDER_PATH):
        # 过滤非图片文件
        if filename.lower().endswith(SUPPORTED_FORMATS):
            # 拆分文件名和后缀
            filename_no_ext, ext = os.path.splitext(filename)
            # 提取学生名字
            student_name = extract_student_name(filename_no_ext)

            if student_name:
                # 生成新文件名（名字+原后缀）
                new_filename = f"{student_name}{ext}"
                # 拼接原路径和新路径
                old_path = os.path.join(FOLDER_PATH, filename)
                new_path = os.path.join(FOLDER_PATH, new_filename)

                # 处理重名（如果已有同名文件，加数字后缀）
                counter = 1
                temp_new_path = new_path
                while os.path.exists(temp_new_path):
                    temp_new_path = os.path.join(FOLDER_PATH, f"{student_name}_{counter}{ext}")
                    counter += 1
                new_path = temp_new_path

                rename_list.append({
                    'old': old_path,
                    'new': new_path,
                    'old_name': filename,
                    'new_name': os.path.basename(new_path)
                })
            else:
                # 文件名格式错误，加入错误列表
                error_list.append(filename)

    # 预览结果
    print("=" * 60)
    if rename_list:
        print(f"📋 待重命名的文件列表（共{len(rename_list)}个）：")
        for idx, item in enumerate(rename_list, 1):
            print(f"{idx:2d}. 原文件名：{item['old_name']} → 新文件名：{item['new_name']}")
    else:
        print("📭 未找到符合格式的图片文件！")

    if error_list:
        print(f"\n❌ 格式错误的文件（共{len(error_list)}个）：")
        for filename in error_list:
            print(f"   - {filename}（请检查是否为「xxx-名字-xxx.后缀」格式）")
    print("=" * 60)

    # 执行重命名（非测试模式下）
    if not TEST_MODE and rename_list:
        confirm = input("\n⚠️  是否确认执行重命名？(输入y/Y确认，其他取消)：")
        if confirm.lower() == 'y':
            success_count = 0
            fail_count = 0
            for item in rename_list:
                try:
                    os.rename(item['old'], item['new'])
                    print(f"✅ 重命名成功：{item['old_name']} → {item['new_name']}")
                    success_count += 1
                except Exception as e:
                    print(f"❌ 重命名失败 {item['old_name']}：{str(e)}")
                    fail_count += 1
            print(f"\n🎉 重命名完成！成功：{success_count} 个，失败：{fail_count} 个")
        else:
            print("🚫 用户取消了重命名操作")
    elif TEST_MODE:
        print("\nℹ️ 当前为【测试模式】，未实际修改文件名！确认预览结果后，将TEST_MODE改为False重新运行。")


if __name__ == "__main__":
    batch_rename_images()