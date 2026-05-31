# 国赛 C 题支撑材料归档

本目录用于归档以往比赛的题目材料、数据、分析代码和图表结果。仅做目录整理。

## 目录结构

```text
.
├── 题目/                 # 题目 PDF、附件和相关说明材料
├── solution/             # 各问求解代码
│   ├── data.py           # 男胎数据清洗与探索
│   ├── Q1.py             # 问题一分析
│   ├── Q2.py             # 问题二建模
│   ├── Q3.py             # 问题三建模
│   └── Q4.py             # 问题四女胎异常判定
├── data/
│   ├── raw/              # 原始 CSV 数据
│   └── processed/        # 清洗后的 CSV 数据
├── figure/               # 归档图表
├── requirements.txt      # Python 依赖
└── README.md
```

## 环境依赖

```bash
pip install -r requirements.txt
```

常用脚本含义：

- `data.py`：读取 `男胎.csv`，生成 `男胎_cleaned.csv`
- `Q1.py`、`Q2.py`、`Q3.py`：读取 `男胎_cleaned.csv`
- `Q4.py`：读取 `女胎.csv`，生成女胎异常判定分析结果
