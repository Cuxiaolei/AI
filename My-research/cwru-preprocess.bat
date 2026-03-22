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
python -m data.preprocess_cwru --configs data/config/cwru/cwru_T1_300-1.yaml
python -m data.preprocess_cwru --configs data/config/cwru/cwru_T1_300-5.yaml
python -m data.preprocess_cwru --configs data/config/cwru/cwru_T1_300-10.yaml
python -m data.preprocess_cwru --configs data/config/cwru/cwru_T1_300-20.yaml

# 任务2 系列（fault_per_class_per_domain = 1/5/10/20）
python -m data.preprocess_cwru --configs data/config/cwru/cwru_T2_300-1.yaml
python -m data.preprocess_cwru --configs data/config/cwru/cwru_T2_300-5.yaml
python -m data.preprocess_cwru --configs data/config/cwru/cwru_T2_300-10.yaml
python -m data.preprocess_cwru --configs data/config/cwru/cwru_T2_300-20.yaml

# 任务3 系列（fault_per_class_per_domain = 1/5/10/20）
python -m data.preprocess_cwru --configs data/config/cwru/cwru_T3_300-1.yaml
python -m data.preprocess_cwru --configs data/config/cwru/cwru_T3_300-5.yaml
python -m data.preprocess_cwru --configs data/config/cwru/cwru_T3_300-10.yaml
python -m data.preprocess_cwru --configs data/config/cwru/cwru_T3_300-20.yaml

# 任务4 系列（fault_per_class_per_domain = 1/5/10/20）
python -m data.preprocess_cwru --configs data/config/cwru/cwru_T4_300-1.yaml
python -m data.preprocess_cwru --configs data/config/cwru/cwru_T4_300-5.yaml
python -m data.preprocess_cwru --configs data/config/cwru/cwru_T4_300-10.yaml
python -m data.preprocess_cwru --configs data/config/cwru/cwru_T4_300-20.yaml