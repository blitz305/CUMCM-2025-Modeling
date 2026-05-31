import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.stats import shapiro
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib as mpl

# 设置Matplotlib正常显示中文和负号
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':

    try:
        df_male = pd.read_csv('男胎_cleaned.csv', encoding='utf-8-sig')
    except FileNotFoundError:
        print("未找到'男胎_cleaned.csv'文件")
        exit()
    print(f"用于分析的有效数据共 {len(df_male)} 条。")

    # 正态性检验
    normality_test_vars = ['Y染色体浓度', '检测孕周_周数', '孕妇BMI', '年龄']
    results_normality = {}
    for var in normality_test_vars:
        # 如果样本量大于5000，则抽样检验以避免shapiro检验的限制
        if len(df_male[var]) > 5000:
            data_to_test = df_male[var].sample(5000, random_state=42)
        else:
            data_to_test = df_male[var]

        stat, p_value = shapiro(data_to_test)
        results_normality[var] = {'W-statistic': stat, 'p-value': p_value}

    for var, result in results_normality.items():
        print(f"变量: {var:<15} | W统计量: {result['W-statistic']:.4f}, p值: {result['p-value']:.4e} ", end="")
        if result['p-value'] < 0.05:
            print(" -> 数据不服从正态分布。")
        else:
            print(" -> 数据可视为服从正态分布。")

    # Spearman相关性分析
    corr_features = ['Y染色体浓度', '检测孕周_周数', '孕妇BMI', '年龄']
    df_corr = df_male[corr_features]
    corr_matrix, p_matrix = spearmanr(df_corr)

    print("\nSpearman相关系数矩阵:")
    print(pd.DataFrame(corr_matrix, index=df_corr.columns, columns=df_corr.columns).round(4))
    print("\nP值矩阵:")
    print(pd.DataFrame(p_matrix, index=df_corr.columns, columns=df_corr.columns).round(4))

    # 多元线性回归分析
    X = df_male[['检测孕周_周数', '孕妇BMI', '年龄']]
    y = df_male['Y染色体浓度']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # 模型检验
    residuals = model.resid
    predicted_values = model.predict(X)

    # 残差Q-Q图
    fig = sm.qqplot(residuals, line='s')
    plt.title('残差的正态性检验 (Q-Q图)')
    plt.show()

    # 残差与拟合值图
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=predicted_values, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('同方差性检验 (残差 vs 拟合值)')
    plt.xlabel('拟合值')
    plt.ylabel('残差')
    plt.show()

    # 多重共线性检验 (VIF)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("\n多重共线性检验 (VIF值):")
    print(vif_data)

    # Bootstrapping检验模型稳定性
    n_bootstraps = 1000
    bootstrapped_coeffs = []
    for _ in range(n_bootstraps):
        sample_df = df_male.sample(n=len(df_male), replace=True)
        X_boot = sample_df[['检测孕周_周数', '孕妇BMI', '年龄']]
        y_boot = sample_df['Y染色体浓度']
        X_boot = sm.add_constant(X_boot)
        model_boot = sm.OLS(y_boot, X_boot).fit()
        bootstrapped_coeffs.append(model_boot.params)

    bootstrapped_coeffs_df = pd.DataFrame(bootstrapped_coeffs)

    bootstrapped_coeffs_df.hist(bins=30, figsize=(12, 8), layout=(2, 2))
    plt.suptitle('Bootstrapping回归系数分布图')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()