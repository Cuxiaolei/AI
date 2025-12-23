import pandas as pd
import traceback
import os


def force_score_convert(score):
    """强制转换总成绩为数字"""
    if pd.isna(score):
        return 0.0
    if isinstance(score, str):
        score_clean = score.strip()
        if score_clean == "不及格":
            return 0.0
        elif score_clean in ["优", "良", "中", "及格"]:
            return 100.0
        try:
            return float(score_clean)
        except:
            return 100.0
    return float(score)


def judge_fail_status(group):
    """极简挂科判定：所有成绩<60 → 挂科"""
    group['总成绩_数字'] = group['总成绩'].apply(force_score_convert)
    all_scores = group['总成绩_数字'].tolist()
    all_below_60 = all(score < 60 for score in all_scores)
    return all_below_60


def process_single_file(file_name, output_dir="."):
    input_path = os.path.join(output_dir, file_name)
    if not os.path.exists(input_path):
        print(f"⚠️ 文件不存在，跳过：{input_path}")
        return None

    file_prefix = os.path.splitext(file_name)[0]
    output_path = os.path.join(output_dir, f"{file_prefix}学生挂科统计结果.xlsx")

    try:
        # 1. 读取数据
        df = pd.read_excel(input_path, engine='openpyxl')
        print(f"\n========== 处理文件：{file_name} ==========")

        # 2. 关键：学号强制标准化为字符串（解决类型不匹配）
        df['学号'] = df['学号'].astype(str).str.strip()  # 转字符串+去空格
        # 处理学号可能的科学计数法/空值
        df['学号'] = df['学号'].replace('nan', '').replace('', '未知学号')

        # 3. 课程名称标准化
        df['课程名称'] = df['课程名称'].astype(str).str.strip()
        df = df.fillna({'补重学期': '', '考试性质': '', '总成绩': 0})

        # 4. 按学号+课程分组，筛选挂科课程
        grouped = df.groupby(['学号', '课程名称'], dropna=False)
        # 直接生成挂科课程的DataFrame（避免后续merge问题）
        fail_mask = grouped.apply(judge_fail_status).reset_index(name='是否挂科')
        fail_courses = fail_mask[fail_mask['是否挂科'] == True][['学号', '课程名称']]

        # 5. 调试：打印挂科课程和学生信息的学号
        print(f"\n🔍 挂科课程的学号列表：{fail_courses['学号'].unique().tolist()}")
        print(f"🔍 所有学生的学号列表：{df['学号'].unique().tolist()}")

        # 6. 统计挂科数目和科目（直接基于df的学号分组，避免merge）
        def get_fail_info(student_df):
            """给单个学生的DataFrame返回挂科数目和科目"""
            student_id = student_df['学号'].iloc[0]
            # 筛选该学生的挂科课程
            student_fail = fail_courses[fail_courses['学号'] == student_id]
            if len(student_fail) > 0:
                return pd.Series([len(student_fail), ','.join(student_fail['课程名称'])])
            else:
                return pd.Series([0, '无'])

        # 7. 按学号分组生成最终结果（一步到位，避免merge）
        result_df = df.groupby('学号').agg({
            '姓名': 'first',
            '班级名称': lambda x: x.mode()[0].strip() if not x.mode().empty else ''
        }).reset_index()

        # 8. 新增挂科数目和科目（核心修复：直接关联）
        result_df[['挂科数目', '挂科科目']] = result_df.apply(
            lambda row: get_fail_info(df[df['学号'] == row['学号']]),
            axis=1
        )

        # 9. 数据类型修正
        result_df['挂科数目'] = result_df['挂科数目'].astype(int)

        # 10. 调试+导出
        print(f"\n✅ 最终统计结果：")
        print(result_df.to_string(index=False))

        result_df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"\n✅ 文件已导出：{output_path}")

        return result_df

    except Exception as e:
        print(f"❌ 处理失败：{str(e)}")
        traceback.print_exc()
        return None


def batch_process_files():
    target_files = [
        "22计科.xlsx", "22软工.xlsx", "22网工.xlsx",
        "22物联网.xlsx", "22智科.xlsx", "22人工.xlsx",
        "23智科.xlsx", "23人工.xlsx", "23网工.xlsx"
    ]

    all_results = []
    for file_name in target_files:
        single_result = process_single_file(file_name)
        if single_result is not None:
            all_results.append(single_result)

    if all_results:
        summary_df = pd.concat(all_results, ignore_index=True)
        summary_path = "所有班级挂科统计汇总.xlsx"
        summary_df.to_excel(summary_path, index=False, engine='openpyxl')
        print(f"\n📊 汇总文件已生成：{summary_path}")


if __name__ == '__main__':
    batch_process_files()
    print("\n✅ 所有处理完成！")