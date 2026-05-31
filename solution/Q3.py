import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from scipy.stats import norm
from sklearn.metrics import adjusted_rand_score
import warnings
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=FutureWarning)

def get_xgboost_model_q3(df):
    """构建问题三的XGBoost回归模型。"""
    features = ['检测孕周_周数', '孕妇BMI', '年龄', '身高', '体重']
    X = df[features]
    y = df['Y染色体浓度']

    class DummyScaler:
        def transform(self, data):
            return data
    scaler = DummyScaler()

    model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=100, max_depth=5,
        learning_rate=0.1, random_state=42
    )
    model.fit(X, y)

    residuals = y - model.predict(X)
    sigma_err = residuals.std()
    return model, scaler, sigma_err, features

def perform_clustering_q3(df):
    """对孕妇进行多维特征聚类。"""
    clustering_features = ['孕妇BMI', '年龄', '身高', '体重']
    cluster_data = df[clustering_features]
    cluster_scaler = StandardScaler()
    cluster_data_scaled = cluster_scaler.fit_transform(cluster_data)

    sse = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(cluster_data_scaled)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, 'bo-')
    plt.title('肘部法则确定最佳K值 (多维特征)')
    plt.xlabel('聚类数量 (K)')
    plt.ylabel('簇内误差平方和 (SSE)')
    plt.grid(True)
    plt.show()

    OPTIMAL_K = 4
    kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
    df['Multi_Factor_Group'] = kmeans.fit_predict(cluster_data_scaled)

    group_summary = df.groupby('Multi_Factor_Group')[clustering_features].mean()
    group_summary['样本数'] = df.groupby('Multi_Factor_Group').size()
    group_summary = group_summary.sort_values('孕妇BMI').reset_index()

    return df, group_summary, cluster_data_scaled, OPTIMAL_K

TARGET_SUCCESS_RATIO = 0.95

def calculate_group_success_ratio(t, group_df, model, scaler, sigma_err, features):
    """计算在给定时间t，一个孕妇群体的平均检测成功率。"""
    group_features_df = group_df[features].copy()
    group_features_df['检测孕周_周数'] = t
    group_features_scaled = scaler.transform(group_features_df)
    predicted_means = model.predict(group_features_scaled)
    TARGET_CONCENTRATION = 0.04
    z_scores = (predicted_means - TARGET_CONCENTRATION) / sigma_err
    return np.mean(norm.cdf(z_scores))

def new_objective_function(t, group_df, model, scaler, sigma_err, features, target_ratio):
    """定义目标函数：时间成本+未达标惩罚。"""
    time_penalty = t
    current_ratio = calculate_group_success_ratio(t, group_df, model, scaler, sigma_err, features)
    success_penalty = 10000 if current_ratio < target_ratio else 0
    return time_penalty + success_penalty

if __name__ == '__main__':
    try:
        df_male = pd.read_csv('男胎_cleaned.csv', encoding='utf-8-sig')
    except FileNotFoundError:
        print("错误：未找到'男胎_cleaned.csv'文件。")
        exit()
    print(f"成功加载清洗后的数据，共 {len(df_male)} 条记录。")

    multi_model, multi_scaler, multi_sigma_err, features_list = get_xgboost_model_q3(df_male)
    print(f"XGBoost模型残差标准差 (σ'_err) 估计为: {multi_sigma_err:.4f}")

    df_male, group_summary, cluster_data_scaled, OPTIMAL_K = perform_clustering_q3(df_male)
    print("\n多维特征分组均值:")
    print(group_summary)
    print(f"\n目标群体达标比例: {TARGET_SUCCESS_RATIO:.0%}")

    # 优化计算
    time_grid = np.arange(10.0, 28.5, 0.5)
    results = []
    plt.figure(figsize=(14, 8))
    for index, row in group_summary.iterrows():
        group_id = row['Multi_Factor_Group']
        current_group_df = df_male[df_male['Multi_Factor_Group'] == group_id]
        objective_values = [
            new_objective_function(t, current_group_df, multi_model, multi_scaler, multi_sigma_err, features_list,
                                   TARGET_SUCCESS_RATIO) for t in time_grid]
        min_obj_index = np.argmin(objective_values)
        best_time = time_grid[min_obj_index]
        success_ratios_over_time = [
            calculate_group_success_ratio(t, current_group_df, multi_model, multi_scaler, multi_sigma_err,
                                          features_list) for t in time_grid]
        results.append({
            'Multi_Factor_Group': group_id,
            '最佳检测时点(周)': best_time if min(objective_values) < 10000 else "无法满足要求",
            '在最佳时点的达标比例': success_ratios_over_time[min_obj_index] if min(objective_values) < 10000 else "N/A"
        })
        group_mean_bmi = row['孕妇BMI']
        plt.plot(time_grid, success_ratios_over_time, marker='.', linestyle='-',
                 label=f'分组 {index + 1} (均值BMI: {group_mean_bmi:.2f})')

    result_df = pd.DataFrame(results)
    final_summary_df = pd.merge(group_summary, result_df, on='Multi_Factor_Group')
    final_summary_df.rename(columns={'Multi_Factor_Group': '分组ID'}, inplace=True)
    print("\n各分组优化结果:")
    print(final_summary_df.round(2))

    plt.axhline(y=TARGET_SUCCESS_RATIO, color='r', linestyle='--', label=f'{TARGET_SUCCESS_RATIO:.0%} 目标线')
    plt.xlabel('检测孕周 (周)')
    plt.ylabel('群体达标比例')
    plt.title('各分组群体达标比例随时间变化')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.show()

    # 聚类稳定性检验
    print("\n多维聚类稳定性检验 (Bootstrap + ARI):")
    base_labels = df_male['Multi_Factor_Group']
    n_bootstraps = 100
    ari_scores = []
    for i in range(n_bootstraps):
        boot_indices = np.random.choice(len(df_male), size=len(df_male), replace=True)
        boot_data_scaled = cluster_data_scaled[boot_indices]
        kmeans_boot = KMeans(n_clusters=OPTIMAL_K, random_state=i, n_init=10)
        boot_labels_on_boot_data = kmeans_boot.fit_predict(boot_data_scaled)
        base_labels_on_boot_data = base_labels.iloc[boot_indices].values
        ari = adjusted_rand_score(base_labels_on_boot_data, boot_labels_on_boot_data)
        ari_scores.append(ari)
    print(f"平均调整兰德系数(ARI): {np.mean(ari_scores):.4f} (标准差: {np.std(ari_scores):.4f})")

    # 可视化推荐时点
    valid_results_df = final_summary_df[final_summary_df['最佳检测时点(周)'] != "无法满足要求"].copy()
    if not valid_results_df.empty:
        valid_results_df['最佳检测时点(周)'] = valid_results_df['最佳检测时点(周)'].astype(float)
        valid_results_df['分组'] = valid_results_df.index.map(lambda i: f"分组 {i+1}")

        plt.figure(figsize=(10, 6))
        bars = sns.barplot(x='分组', y='最佳检测时点(周)', data=valid_results_df, palette='viridis')
        for bar in bars.patches:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.1f} 周',
                     va='bottom', ha='center', fontsize=12, color='black')
        plt.title('各孕妇分组的最佳NIPT时点推荐', fontsize=16)
        plt.xlabel('孕妇分组 (按均值BMI升序)', fontsize=12)
        plt.ylabel('推荐的最早检测时点 (周)', fontsize=12)
        plt.ylim(0, max(valid_results_df['最佳检测时点(周)']) * 1.2)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    else:
        print("所有分组均无法在指定时间内满足目标达标率，无法生成推荐时点图。")

    # 特征重要性分析
    feature_importance = pd.DataFrame({
        '特征': features_list,
        '重要性': multi_model.feature_importances_
    }).sort_values('重要性', ascending=False)
    plt.figure(figsize=(12, 7))
    sns.barplot(x='重要性', y='特征', data=feature_importance, palette='rocket')
    plt.title('XGBoost模型特征重要性', fontsize=16)
    plt.xlabel('重要性得分 (Gain)', fontsize=12)
    plt.ylabel('特征', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

    # 检测误差敏感性分析
    print("\n检测误差影响分析:")
    error_scenarios_q3 = {
        f"低误差(σ'*0.7)": multi_sigma_err * 0.7,
        f"基准(σ')": multi_sigma_err,
        f"高误差(σ'*1.3)": multi_sigma_err * 1.3,
    }
    error_sensitivity_results_q3 = []
    for scenario_name, current_sigma_err in error_scenarios_q3.items():
        best_times_q3 = []
        for index, row in group_summary.iterrows():
            group_id = row['Multi_Factor_Group']
            current_group_df = df_male[df_male['Multi_Factor_Group'] == group_id]
            objective_values = [
                new_objective_function(t, current_group_df, multi_model, multi_scaler, current_sigma_err, features_list,
                                       TARGET_SUCCESS_RATIO) for t in time_grid
            ]
            min_obj_index = np.argmin(objective_values)
            best_time = time_grid[min_obj_index]
            if min(objective_values) >= 10000:
                best_times_q3.append(np.nan)
            else:
                best_times_q3.append(best_time)
        row_data = {"情景": scenario_name}
        for i, time in enumerate(best_times_q3):
            row_data[f"分组{i + 1}时点"] = time
        error_sensitivity_results_q3.append(row_data)
    error_sensitivity_df_q3 = pd.DataFrame(error_sensitivity_results_q3)
    print(error_sensitivity_df_q3)

    error_sensitivity_df_q3_long = error_sensitivity_df_q3.melt(id_vars='情景', var_name='分组', value_name='最佳时点')
    plt.figure(figsize=(12, 7))
    sns.pointplot(data=error_sensitivity_df_q3_long, x='分组', y='最佳时点', hue='情景',
                  markers=['v', 'o', '^'], linestyles=['--', '-', ':'])
    plt.title('检测误差对最佳时点推荐的影响', fontsize=16)
    plt.xlabel('孕妇分组', fontsize=12)
    plt.ylabel('推荐的最早检测时点 (周)', fontsize=12)
    plt.legend(title='检测误差情景')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()