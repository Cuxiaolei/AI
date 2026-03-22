@echo off
:: 强制切换编码为 UTF-8，并重定向输出避免干扰
chcp 65001 >nul 2>&1
:: 增加短暂延迟，确保编码切换生效
timeout /t 1 /nobreak >nul

:: 设置控制台输出格式（可选，增强兼容性）
set PYTHONIOENCODING=utf-8
set CONSOLE_FONT_CODEPAGE=65001

echo ==================== 开始执行指令 ====================
echo 执行时间：%date% %time%


# 任务5 系列（fault_per_class_per_domain = 1/5/10/20）
python -m preprocess.preprocess_pu --configs preprocess/config/pu/pu_T5_1.yaml
python -m preprocess.preprocess_pu --configs preprocess/config/pu/pu_T5_5.yaml
python -m preprocess.preprocess_pu --configs preprocess/config/pu/pu_T5_10.yaml
python -m preprocess.preprocess_pu --configs preprocess/config/pu/pu_T5_20.yaml


# 任务6 系列（fault_per_class_per_domain = 1/5/10/20）
python -m preprocess.preprocess_pu --configs preprocess/config/pu/pu_T6_1.yaml
python -m preprocess.preprocess_pu --configs preprocess/config/pu/pu_T6_5.yaml
python -m preprocess.preprocess_pu --configs preprocess/config/pu/pu_T6_10.yaml
python -m preprocess.preprocess_pu --configs preprocess/config/pu/pu_T6_20.yaml

# 任务7 系列（fault_per_class_per_domain = 1/5/10/20）
python -m preprocess.preprocess_pu --configs preprocess/config/pu/pu_T7_1.yaml
python -m preprocess.preprocess_pu --configs preprocess/config/pu/pu_T7_5.yaml
python -m preprocess.preprocess_pu --configs preprocess/config/pu/pu_T7_10.yaml
python -m preprocess.preprocess_pu --configs preprocess/config/pu/pu_T7_20.yaml

# 任务8 系列（fault_per_class_per_domain = 1/5/10/20）
python -m preprocess.preprocess_pu --configs preprocess/config/pu/pu_T8_1.yaml
python -m preprocess.preprocess_pu --configs preprocess/config/pu/pu_T8_5.yaml
python -m preprocess.preprocess_pu --configs preprocess/config/pu/pu_T8_10.yaml
python -m preprocess.preprocess_pu --configs preprocess/config/pu/pu_T8_20.yaml