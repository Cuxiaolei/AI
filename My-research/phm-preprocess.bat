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

# 任务1 系列（fault_per_class_per_domain = 1/5/10/20）
python -m preprocess.preprocess_phm-spur --configs preprocess/config/phm/phm-spur_T1_1.yaml
python -m preprocess.preprocess_phm-spur --configs preprocess/config/phm/phm-spur_T1_5.yaml
python -m preprocess.preprocess_phm-spur --configs preprocess/config/phm/phm-spur_T1_10.yaml
python -m preprocess.preprocess_phm-spur --configs preprocess/config/phm/phm-spur_T1_20.yaml

# 任务6 系列（fault_per_class_per_domain = 1/5/10/20）
python -m preprocess.preprocess_phm-spur --configs preprocess/config/phm/phm-spur_T2_1.yaml
python -m preprocess.preprocess_phm-spur --configs preprocess/config/phm/phm-spur_T2_5.yaml
python -m preprocess.preprocess_phm-spur --configs preprocess/config/phm/phm-spur_T2_10.yaml
python -m preprocess.preprocess_phm-spur --configs preprocess/config/phm/phm-spur_T2_20.yaml

# 任务7 系列（fault_per_class_per_domain = 1/5/10/20）
python -m preprocess.preprocess_phm-spur --configs preprocess/config/phm/phm-spur_T3_1.yaml
python -m preprocess.preprocess_phm-spur --configs preprocess/config/phm/phm-spur_T3_5.yaml
python -m preprocess.preprocess_phm-spur --configs preprocess/config/phm/phm-spur_T3_10.yaml
python -m preprocess.preprocess_phm-spur --configs preprocess/config/phm/phm-spur_T3_20.yaml

# 任务8 系列（fault_per_class_per_domain = 1/5/10/20）
python -m preprocess.preprocess_phm-spur --configs preprocess/config/phm/phm-spur_T4_1.yaml
python -m preprocess.preprocess_phm-spur --configs preprocess/config/phm/phm-spur_T4_5.yaml
python -m preprocess.preprocess_phm-spur --configs preprocess/config/phm/phm-spur_T4_10.yaml
python -m preprocess.preprocess_phm-spur --configs preprocess/config/phm/phm-spur_T4_20.yaml

