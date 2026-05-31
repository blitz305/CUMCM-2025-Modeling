import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy.stats import norm
import warnings
import matplotlib as mpl
from sklearn.metrics import adjusted_rand_score
import xgboost as xgb


# --- 环境设置 ---
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=FutureWarning)


def get_regression_model(df):
    """基于XGBoost训练回归模型并估算残差标准差。"""
    features = ['检测孕周_周数', '孕妇BMI']
    X = df[features]
    y = df['Y染色体浓度']

    # 使用一个虚拟的Scaler，因为XGBoost不需要标准化
    class DummyScaler:
        def transform(self, data):
            return data

    scaler = DummyScaler()

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)

    residuals = y - model.predict(X)
    sigma_err = residuals.std()

    return model, scaler, sigma_err


def time_penalty(t, w_mid=1, w_late=10):
    """计算时间惩罚函数。"""
    if t <= 12:
        return 0
    elif 12 < t <= 27:
        return w_mid * (t - 12)
    else:
        return w_mid * (27 - 12) + w_late * (t - 27)


def failure_probability(t, group_mean_bmi, model, scaler, sigma_err):
    """计算在给定时间t的检测失败概率。"""
    features_df = pd.DataFrame([[t, group_mean_bmi]], columns=['检测孕周_周数', '孕妇BMI'])
    features_scaled = scaler.transform(features_df)
    predicted_mean_c = model.predict(features_scaled)[0]
    TARGET_CONCENTRATION = 0.04
    z_score = (TARGET_CONCENTRATION - predicted_mean_c) / sigma_err
    return norm.cdf(z_score)


def total_risk(t, group_mean_bmi, model, scaler, sigma_err, w1=0.5, w2=0.5):
    """计算总潜在风险。"""
    p_t = time_penalty(t)
    f_t = failure_probability(t, group_mean_bmi, model, scaler, sigma_err)
    return w1 * p_t + w2 * f_t


if __name__ == '__main__':
    try:
        df_male = pd.read_csv('男胎_cleaned.csv', encoding='utf-8')
    except FileNotFoundError:
        print("错误：未找到'男胎_cleaned.csv'文件。")
        exit()
    print(f"成功加载清洗后的数据，共 {len(df_male)} 条记录。")

    model, scaler, sigma_err = get_regression_model(df_male)
    print(f"预测模型的残差标准差 (σ_err) 估计为: {sigma_err:.4f}")

    # K-Means聚类确定分组
    bmi_data = df_male[['孕妇BMI']]
    sse = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(bmi_data)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, 'bo-')
    plt.title('肘部法则确定最佳K值')
    plt.xlabel('聚类数量 (K)')
    plt.ylabel('簇内误差平方和 (SSE)')
    plt.grid(True)
    plt.show()

    OPTIMAL_K = 4
    kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
    df_male['BMI_Group'] = kmeans.fit_predict(bmi_data)

    group_summary = df_male.groupby('BMI_Group')['孕妇BMI'].agg(['mean', 'min', 'max', 'count'])
    group_summary.columns = ['BMI均值', 'BMI最小值', 'BMI最大值', '样本数']
    group_summary = group_summary.sort_values('BMI均值').reset_index()
    group_summary['有序分组ID'] = group_summary.index
    mapping = group_summary.set_index('BMI_Group')['有序分组ID']
    df_male['BMI_Group_Ordered'] = df_male['BMI_Group'].map(mapping)
    print("\nBMI分组特征:")
    print(group_summary)

    # 计算并可视化各组最佳时点
    time_grid = np.arange(10.0, 28.5, 0.5)
    results = []
    plt.figure(figsize=(14, 8))

    for group_id in group_summary['有序分组ID']:
        current_group_info = group_summary[group_summary['有序分组ID'] == group_id]
        group_mean_bmi = current_group_info['BMI均值'].iloc[0]
        risks = [total_risk(t, group_mean_bmi, model, scaler, sigma_err) for t in time_grid]
        min_risk_index = np.argmin(risks)
        best_time = time_grid[min_risk_index]
        min_risk = risks[min_risk_index]
        results.append({
            '分组': f"第 {group_id + 1} 组",
            'BMI区间': f"[{current_group_info['BMI最小值'].iloc[0]:.2f}, {current_group_info['BMI最大值'].iloc[0]:.2f}]",
            '最佳检测时点(周)': best_time,
            '最低潜在风险': min_risk
        })
        plt.plot(time_grid, risks, marker='.', linestyle='-',
                 label=f'分组 {group_id + 1} (BMI均值: {group_mean_bmi:.2f})')

    result_df = pd.DataFrame(results)
    print("\n各分组最佳NIPT时点优化结果:")
    print(result_df)

    plt.title('各BMI分组的潜在风险随检测时间变化曲线')
    plt.xlabel('检测孕周 (周)')
    plt.ylabel('潜在风险值 (无量纲)')
    plt.axvline(x=12, color='gray', linestyle='--', label='早期/中期分界')
    plt.axvline(x=28, color='red', linestyle='--', label='中期/晚期分界')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x='分组', y='最佳检测时点(周)', data=result_df, palette='viridis')
    for bar in bars.patches:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.1f} 周',
                 va='bottom', ha='center', fontsize=12, color='black')
    plt.title('各BMI分组的最佳NIPT检测时点推荐', fontsize=16)
    plt.xlabel('孕妇分组 (按BMI升序)', fontsize=12)
    plt.ylabel('推荐检测时点 (周)', fontsize=12)
    plt.ylim(0, max(result_df['最佳检测时点(周)']) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    plt.figure(figsize=(12, 7))
    sns.histplot(df_male['孕妇BMI'], kde=True, bins=40, label='全体孕妇BMI分布')
    colors = ['red', 'green', 'purple', 'orange']
    for i in range(OPTIMAL_K):
        group_info = group_summary[group_summary['有序分组ID'] == i]
        min_bmi = group_info['BMI最小值'].iloc[0]
        max_bmi = group_info['BMI最大值'].iloc[0]
        plt.axvspan(min_bmi, max_bmi, color=colors[i], alpha=0.2, label=f'分组 {i + 1}: [{min_bmi:.1f}, {max_bmi:.1f}]')
    plt.title('孕妇BMI分布与K-Means聚类结果', fontsize=16)
    plt.xlabel('孕妇BMI', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x='分组', y='最低潜在风险', data=result_df, palette='coolwarm')
    for bar in bars.patches:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.4f}',
                 va='bottom', ha='center', fontsize=12, color='black')
    plt.title('各BMI分组在最佳时点的最低潜在风险', fontsize=16)
    plt.xlabel('孕妇分组 (按BMI升序)', fontsize=12)
    plt.ylabel('最低潜在风险值 (无量纲)', fontsize=12)
    plt.ylim(0, max(result_df['最低潜在风险']) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # 检测误差的参数敏感性分析
    error_scenarios = {
        f"低误差(σ*0.7)": sigma_err * 0.7,
        f"基准(σ)": sigma_err,
        f"高误差(σ*1.3)": sigma_err * 1.3,
    }
    error_sensitivity_results = []
    for scenario_name, current_sigma_err in error_scenarios.items():
        best_times = []
        for index, row in group_summary.iterrows():
            group_mean_bmi = row['BMI均值']
            risks = [total_risk(t, group_mean_bmi, model, scaler, current_sigma_err) for t in time_grid]
            best_time = time_grid[np.argmin(risks)]
            best_times.append(best_time)
        row_data = {"情景": scenario_name}
        for i, time in enumerate(best_times):
            row_data[f"分组{i + 1}时点"] = time
        error_sensitivity_results.append(row_data)

    error_sensitivity_df = pd.DataFrame(error_sensitivity_results)
    print("\n不同检测误差水平下的最佳时点推荐:")
    print(error_sensitivity_df)

    error_sensitivity_df_long = error_sensitivity_df.melt(id_vars='情景', var_name='分组', value_name='最佳时点')
    plt.figure(figsize=(12, 7))
    sns.pointplot(data=error_sensitivity_df_long, x='分组', y='最佳时点', hue='情景',
                  markers=['v', 'o', '^'], linestyles=['--', '-', ':'])
    plt.title('检测误差对最佳时点推荐的影响', fontsize=16)
    plt.xlabel('孕妇分组', fontsize=12)
    plt.ylabel('最佳检测时点 (周)', fontsize=12)
    plt.legend(title='检测误差情景')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # --- 模型检验 ---
    # 聚类结果鲁棒性检验
    base_labels = df_male['BMI_Group_Ordered']
    n_bootstraps = 100
    ari_scores = []
    for i in range(n_bootstraps):
        boot_indices = np.random.choice(len(df_male), size=len(df_male), replace=True)
        boot_data = bmi_data.iloc[boot_indices]
        kmeans_boot = KMeans(n_clusters=OPTIMAL_K, random_state=i, n_init=10)
        boot_labels_on_boot_data = kmeans_boot.fit_predict(boot_data)
        base_labels_on_boot_data = base_labels.iloc[boot_indices]
        ari = adjusted_rand_score(base_labels_on_boot_data, boot_labels_on_boot_data)
        ari_scores.append(ari)

    print("\n聚类稳定性检验 (Bootstrap + ARI):")
    print(f"平均调整兰德系数(ARI): {np.mean(ari_scores):.4f} (标准差: {np.std(ari_scores):.4f})")

    # 风险权重参数敏感性分析
    scenarios = {
        "基准 (w1=0.5, w2=0.5)": (0.5, 0.5),
        "保守 (w1=0.2, w2=0.8)": (0.2, 0.8),
        "激进 (w1=0.8, w2=0.2)": (0.8, 0.2)
    }
    sensitivity_results = []
    for scenario_name, (w1, w2) in scenarios.items():
        best_times = []
        for index, row in group_summary.iterrows():
            group_mean_bmi = row['BMI均值']
            risks = [total_risk(t, group_mean_bmi, model, scaler, sigma_err, w1=w1, w2=w2) for t in time_grid]
            best_time = time_grid[np.argmin(risks)]
            best_times.append(best_time)
        row_data = {"情景": scenario_name}
        for i, time in enumerate(best_times):
            row_data[f"分组{i + 1}时点"] = time
        sensitivity_results.append(row_data)

    sensitivity_df = pd.DataFrame(sensitivity_results)
    print("\n不同风险权重下的最佳时点推荐:")
    print(sensitivity_df)