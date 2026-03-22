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


# 任务9 系列（fault_per_class_per_domain = 1/5/10/20）
python -m data.preprocess_pu --configs data/config/pu/pu_T9_300-1.yaml
python -m data.preprocess_pu --configs data/config/pu/pu_T9_300-5.yaml
python -m data.preprocess_pu --configs data/config/pu/pu_T9_300-10.yaml
python -m data.preprocess_pu --configs data/config/pu/pu_T9_300-20.yaml

# 任务10 系列（fault_per_class_per_domain = 1/5/10/20）
python -m data.preprocess_pu --configs data/config/pu/pu_T10_300-1.yaml
python -m data.preprocess_pu --configs data/config/pu/pu_T10_300-5.yaml
python -m data.preprocess_pu --configs data/config/pu/pu_T10_300-10.yaml
python -m data.preprocess_pu --configs data/config/pu/pu_T10_300-20.yaml

# 任务11 系列（fault_per_class_per_domain = 1/5/10/20）
python -m data.preprocess_pu --configs data/config/pu/pu_T11_300-1.yaml
python -m data.preprocess_pu --configs data/config/pu/pu_T11_300-5.yaml
python -m data.preprocess_pu --configs data/config/pu/pu_T11_300-10.yaml
python -m data.preprocess_pu --configs data/config/pu/pu_T11_300-20.yaml

# 任务12 系列（fault_per_class_per_domain = 1/5/10/20）
python -m data.preprocess_pu --configs data/config/pu/pu_T12_300-1.yaml
python -m data.preprocess_pu --configs data/config/pu/pu_T12_300-5.yaml
python -m data.preprocess_pu --configs data/config/pu/pu_T12_300-10.yaml
python -m data.preprocess_pu --configs data/config/pu/pu_T12_300-20.yaml