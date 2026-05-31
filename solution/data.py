import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=FutureWarning)

def parse_gestational_week(gw_str):
    """将'12w+3'格式的孕周字符串转换为数值（周数）。"""
    try:
        if isinstance(gw_str, str) and 'w+' in gw_str:
            weeks, days = gw_str.replace('w', '').split('+')
            return float(weeks) + float(days) / 7
        # 处理'12w'这种没有天数的格式
        elif isinstance(gw_str, str) and 'w' in gw_str and '+' not in gw_str:
            return float(gw_str.replace('w', ''))
        return np.nan
    except:
        return np.nan

def clean_and_explore_data(filepath='男胎.csv'):
    """对男胎数据进行清洗和探索性分析。"""
    try:
        df = pd.read_csv(filepath, encoding='gbk')
        print(f"成功加载 '{filepath}'，原始数据共 {len(df)} 条记录。")
    except FileNotFoundError:
        print(f"错误：未找到'{filepath}'文件。")
        return None

    df['检测孕周_周数'] = df['检测孕周'].apply(parse_gestational_week)

    # 基于GC含量进行数据过滤
    df = df[(df['GC含量'] >= 0.35) & (df['GC含量'] <= 0.65)].copy()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # 对同一孕妇在同一孕周的多次检测记录取均值
    df_cleaned = df.groupby(['孕妇代码', '检测孕周_周数'], as_index=False)[numeric_cols].mean()
    print(f"处理重复检测后，数据压缩至 {len(df_cleaned)} 条。")

    # 描述性统计
    key_vars = ['检测孕周_周数', '孕妇BMI', '年龄', 'Y染色体浓度']
    print(df_cleaned[key_vars].describe().round(2))

    # 变量分布可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('关键变量分布直方图与核密度估计', fontsize=16)
    for ax, var in zip(axes.flatten(), key_vars):
        sns.histplot(df_cleaned[var], kde=True, ax=ax, bins=30)
        ax.set_title(f'{var} 的分布')
        ax.set_ylabel('频数')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Spearman相关性分析
    corr_vars = ['Y染色体浓度', '检测孕周_周数', '孕妇BMI', '年龄']
    plt.figure(figsize=(10, 8))
    corr_matrix = df_cleaned[corr_vars].corr(method='spearman')
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('关键变量Spearman相关性热力图')
    plt.show()

    # 散点图与回归线
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.regplot(ax=axes[0, 0], x='检测孕周_周数', y='Y染色体浓度', data=df_cleaned,
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    axes[0, 0].set_title('Y染色体浓度 vs 检测孕周')
    axes[0, 0].grid(True)

    sns.regplot(ax=axes[0, 1], x='孕妇BMI', y='Y染色体浓度', data=df_cleaned,
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    axes[0, 1].set_title('Y染色体浓度 vs 孕妇BMI')
    axes[0, 1].grid(True)

    sns.regplot(ax=axes[1, 0], x='年龄', y='Y染色体浓度', data=df_cleaned,
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    axes[1, 0].set_title('Y染色体浓度 vs 年龄')
    axes[1, 0].grid(True)

    sns.regplot(ax=axes[1, 1], x='年龄', y='检测孕周_周数', data=df_cleaned,
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'blue'})
    axes[1, 1].set_title('检测孕周 vs 年龄')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # 保存清洗后的数据
    output_filepath = '男胎_cleaned.csv'
    df_cleaned.to_csv(output_filepath, index=False, encoding='utf-8-sig')
    print(f"\n处理后的数据已保存至 '{output_filepath}'")

    return df_cleaned

if __name__ == '__main__':
    cleaned_data = clean_and_explore_data('男胎.csv')
