


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE




SMOTE_AVAILABLE = True

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

if not os.path.exists('Q4results'):
    os.makedirs('Q4results')
    print("已创建Q4results文件夹用于保存输出")


class NIPTDataProcessor:


    def __init__(self):

        self.female_records = None  # 女胎数据

    def load_dataset(self, file_path):

        try:
            # 从CSV文件读取女胎数据
            self.female_records = pd.read_csv(file_path)

            # 标准化列名
            std_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                           'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE']

            self.female_records.columns = std_columns[:len(self.female_records.columns)]

            print(f"数据加载成功: 女胎样本{len(self.female_records)}例")
            return True

        except Exception as e:
            print(f"数据加载出错: {e}")
            return False

    def convert_gestational_week(self, week_str):

        if pd.isna(week_str) or week_str == '':
            return np.nan

        try:
            week_str = str(week_str).strip()
            # 处理包含w的格式
            if 'w' in week_str:
                parts = week_str.split('w')
                weeks = float(parts[0])
                # 处理天数
                if '+' in parts[1]:
                    days = float(parts[1].replace('+', ''))
                    return weeks + days / 7.0
                else:
                    return weeks
            else:
                # 直接是数字
                return float(week_str)
        except:
            return np.nan

    def preprocess_female_records(self):
        """女胎数据预处理与质控"""
        if self.female_records is None:
            print("请先调用load_dataset加载数据")
            return None

        # 创建数据副本
        df = self.female_records.copy()

        # 处理孕周格式
        df['J_week'] = df['J'].apply(self.convert_gestational_week)

        # 记录原始数据量
        original_count = len(df)

        # 质量控制筛选
        df = df[(df['P'] >= 0.35) & (df['P'] <= 0.65)]  # GC含量范围


        # 异常标记: AB列非空表示异常
        df['anomaly_flag'] = (~df['AB'].isna()).astype(int)

        # 删除关键变量缺失的样本
        df = df.dropna(subset=['K', 'J_week', 'Q', 'R', 'S', 'T'])

        # 输出处理结果
        print(f"女胎数据质控: {original_count} → {len(df)}条有效记录")
        print(f"异常样本统计: {df['anomaly_flag'].sum()}例 ({df['anomaly_flag'].mean() * 100:.1f}%)")

        # 保存处理后的数据到CSV文件
        output_filename = '女胎_cleaned.csv'
        df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"处理后的数据已保存至: {output_filename}")

        return df.reset_index(drop=True)


class Problem4Solver:
    """问题4求解器：女胎异常判定分析"""


    def __init__(self, data_processor):
        """初始化求解器"""
        self.data_handler = data_processor
        self.trained_models = {}  # 存储训练的模型
        self.feat_scaler = None  # 特征标准化器
        self.feat_names = None  # 特征名称列表
        self.best_cutoff = 0.5  # 最优决策阈值

    def construct_female_features(self, df):
        """构建女胎特征矩阵"""

        feature_matrix = pd.DataFrame()

        # ---- 核心Z值特征 ----
        feature_matrix['Z13'] = df['Q']  # 13号染色体Z值
        feature_matrix['Z18'] = df['R']  # 18号染色体Z值
        feature_matrix['Z21'] = df['S']  # 21号染色体Z值
        feature_matrix['ZX'] = df['T']  # X染色体Z值

        # ---- X染色体浓度 ----
        if 'W' in df.columns:
            feature_matrix['X_concentration'] = df['W'].fillna(0)
        else:
            feature_matrix['X_concentration'] = 0

        # ---- GC含量特征 ----
        feature_matrix['GC_13'] = df['X']  # 13号染色体GC含量
        feature_matrix['GC_18'] = df['Y']  # 18号染色体GC含量
        feature_matrix['GC_21'] = df['Z']  # 21号染色体GC含量
        feature_matrix['GC_overall'] = df['P']  # 整体GC含量

        # ---- 测序质量特征 ----
        feature_matrix['total_reads'] = np.log10(df['L'])  # 总读段数(对数)
        feature_matrix['mapped_ratio'] = df['M']  # 比对比例
        feature_matrix['dup_ratio'] = df['N']  # 重复比例
        feature_matrix['unique_reads'] = np.log10(df['O'])  # 唯一比对读段数
        feature_matrix['filtered_ratio'] = df['AA']  # 过滤比例

        # ---- 临床特征 ----
        feature_matrix['BMI'] = df['K']  # 体质指数
        feature_matrix['gestational_week'] = df['J_week']  # 孕周
        feature_matrix['age'] = df['C']  # 年龄
        feature_matrix['height'] = df['D']  # 身高
        feature_matrix['weight'] = df['E']  # 体重

        # ---- 派生特征 ----
        # Z值衍生特征
        feature_matrix['Z_max'] = feature_matrix[['Z13', 'Z18', 'Z21']].abs().max(axis=1)  # 最大Z值绝对值
        feature_matrix['Z_sum'] = feature_matrix[['Z13', 'Z18', 'Z21']].abs().sum(axis=1)  # Z值绝对值之和
        # GC含量衍生特征
        feature_matrix['GC_variance'] = feature_matrix[['GC_13', 'GC_18', 'GC_21']].var(axis=1)  # GC含量方差
        feature_matrix['GC_mean'] = feature_matrix[['GC_13', 'GC_18', 'GC_21']].mean(axis=1)  # GC含量均值

        # ---- 交互特征 ----
        feature_matrix['BMI_age'] = feature_matrix['BMI'] * feature_matrix['age']  # BMI与年龄交互
        feature_matrix['Z_BMI'] = feature_matrix['Z_max'] * feature_matrix['BMI']  # Z值与BMI交互
        feature_matrix['Z_week'] = feature_matrix['Z_max'] * feature_matrix['gestational_week']  # Z值与孕周交互

        # ---- 质量评分 ----
        feature_matrix['quality_score'] = (
                feature_matrix['mapped_ratio'] * 0.3 +
                (1 - feature_matrix['dup_ratio']) * 0.3 +
                (1 - feature_matrix['filtered_ratio']) * 0.4
        )

        # ---- 传统筛查规则特征 ----
        feature_matrix['high_risk_Z13'] = (feature_matrix['Z13'].abs() >= 3.0).astype(int)
        feature_matrix['high_risk_Z18'] = (feature_matrix['Z18'].abs() >= 3.0).astype(int)
        feature_matrix['high_risk_Z21'] = (feature_matrix['Z21'].abs() >= 3.0).astype(int)
        feature_matrix['very_high_risk'] = (feature_matrix['Z_max'] >= 3.5).astype(int)
        feature_matrix['borderline_risk'] = ((feature_matrix['Z_max'] >= 2.5) &
                                             (feature_matrix['Z_max'] < 3.5)).astype(int)

        # ---- 质量控制指标 ----
        feature_matrix['gc_abnormal'] = ((feature_matrix['GC_overall'] < 0.4) |
                                         (feature_matrix['GC_overall'] > 0.6)).astype(int)
        feature_matrix['low_quality'] = (feature_matrix['quality_score'] < 0.7).astype(int)

        return feature_matrix

    def address_class_imbalance(self, X, y, method='class_weight'):
        """处理类别不平衡问题"""
        print(f"类别分布: 正常样本 {sum(y == 0)}例, 异常样本 {sum(y == 1)}例")

        # 如果有SMOTE且异常样本足够，使用SMOTE过采样
        if SMOTE_AVAILABLE and method == 'smote' and sum(y == 1) > 1:
            try:
                # 根据少数类样本数量调整邻居数
                k_neighbors = min(5, sum(y == 1) - 1)
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                print(f"SMOTE过采样后: 正常样本 {sum(y_resampled == 0)}例, 异常样本 {sum(y_resampled == 1)}例")
                return X_resampled, y_resampled
            except Exception as e:
                print(f"SMOTE过采样失败: {e}，将使用类权重方式替代")
                return X, y
        else:
            # 默认返回原始数据，在模型中使用类权重
            return X, y

    def build_models(self, df):

        print("开始训练女胎异常预测模型...")

        # 准备特征和标签
        feature_matrix = self.construct_female_features(df)
        target_array = df['anomaly_flag'].values

        print(f"特征维度: {len(feature_matrix.columns)}")
        print(f"样本数量: {len(target_array)}")
        print(f"异常比例: {target_array.mean():.3f}")

        # 检查是否有异常样本
        if target_array.sum() == 0:
            print("警告：数据集中没有异常样本，无法训练模型")
            return None, None

        # 判断是否有足够样本进行分层抽样
        if target_array.sum() >= 2 and (len(target_array) - target_array.sum()) >= 2:
            strat = target_array  # 使用标签进行分层
        else:
            strat = None
            print("样本量不足以进行分层抽样")

        # 划分训练集和测试集
        test_ratio = min(0.3, max(0.1, 1.0 / len(target_array)))  # 动态测试集比例
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix, target_array,
            test_size=test_ratio,
            random_state=42,
            stratify=strat
        )

        # 特征标准化
        self.feat_scaler = StandardScaler()
        X_train_scaled = self.feat_scaler.fit_transform(X_train)
        X_test_scaled = self.feat_scaler.transform(X_test)
        self.feat_names = feature_matrix.columns.tolist()

        # 处理类不平衡
        balance_strategy = 'smote' if SMOTE_AVAILABLE and sum(y_train == 1) > 5 else 'class_weight'
        X_train_balanced, y_train_balanced = self.address_class_imbalance(
            X_train_scaled, y_train, balance_strategy
        )

        # ---- 模型1: L1正则化逻辑回归 ----
        print("训练L1逻辑回归模型...")
        l1_logistic = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=0.1,
            random_state=42,
            max_iter=1000,
            class_weight='balanced' if balance_strategy != 'smote' else None
        )
        l1_logistic.fit(X_train_balanced, y_train_balanced)

        # ---- 模型2: 随机森林 ----
        print("训练随机森林模型...")
        rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        rf_classifier.fit(X_train_balanced, y_train_balanced)

        self.trained_models = {
            'logistic_l1': l1_logistic,
            'random_forest': rf_classifier
        }

        # 评估所有模型
        self.assess_models(X_test_scaled, y_test)

        return X_test_scaled, y_test

    def assess_models(self, X_test, y_test):
        """评估模型性能指标"""
        print("\n===== 模型评估结果 =====")
        evaluation_outcomes = []

        # 逐个评估模型
        for model_name, model in self.trained_models.items():
            # 获取预测概率和类别
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            # 计算评估指标
            if len(np.unique(y_test)) > 1:
                # AUC计算
                auc_score = roc_auc_score(y_test, y_prob)

                # 交叉验证(如果样本足够)
                if len(X_test) >= 10:
                    try:
                        cv_folds = min(3, len(X_test) // 3)
                        cv_results = cross_val_score(
                            model, X_test, y_test,
                            cv=cv_folds, scoring='roc_auc'
                        )
                        cv_auc_mean, cv_auc_std = cv_results.mean(), cv_results.std()
                    except Exception:
                        cv_auc_mean, cv_auc_std = auc_score, 0
                else:
                    cv_auc_mean, cv_auc_std = auc_score, 0
            else:
                # 单一类别情况
                auc_score = 0.5
                cv_auc_mean, cv_auc_std = 0.5, 0

            # 计算准确率
            accuracy = (y_pred == y_test).mean()

            # 保存评估结果
            evaluation_outcomes.append({
                'model': model_name,
                'auc': auc_score,
                'accuracy': accuracy,
                'cv_auc_mean': cv_auc_mean,
                'cv_auc_std': cv_auc_std
            })

            # 输出评估结果
            print(f"{model_name} 模型:")
            print(f"  AUC: {auc_score:.3f}")
            print(f"  准确率: {accuracy:.3f}")
            print(f"  交叉验证AUC: {cv_auc_mean:.3f} ± {cv_auc_std:.3f}")
            print()


        self.evaluation_outcomes = pd.DataFrame(evaluation_outcomes)
        return self.evaluation_outcomes

    def adjust_decision_threshold(self, X_test, y_test, cost_ratio=10):

        print("优化模型决策阈值...")


        if len(self.trained_models) == 0 or len(np.unique(y_test)) == 1:
            print("无法优化阈值：模型不可用或测试集为单一类别")
            self.best_cutoff = 0.5
            return 0.5

        # 选择AUC最高的模型
        best_model_name = self.evaluation_outcomes.loc[
            self.evaluation_outcomes['auc'].idxmax(), 'model'
        ]
        best_model = self.trained_models[best_model_name]

        # 获取预测概率
        probabilities = best_model.predict_proba(X_test)[:, 1]

        # 尝试不同阈值
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5  # 默认阈值
        best_score = -np.inf
        threshold_metrics = []

        for thresh in thresholds:
            y_pred_at_threshold = (probabilities >= thresh).astype(int)

            # 检查预测结果是否有效
            if len(np.unique(y_test)) > 1 and len(np.unique(y_pred_at_threshold)) > 1:
                try:
                    # 计算混淆矩阵
                    cm = confusion_matrix(y_test, y_pred_at_threshold)
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()

                        # 计算关键指标
                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

                        # 考虑成本的综合评分（假阴成本更高）
                        score = sensitivity * 0.7 + specificity * 0.3 - cost_ratio * fn / len(y_test)

                        # 保存阈值结果
                        threshold_metrics.append({
                            'threshold': thresh,
                            'sensitivity': sensitivity,
                            'specificity': specificity,
                            'precision': precision,
                            'cost_score': score,
                            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
                        })

                        # 更新最佳阈值
                        if score > best_score:
                            best_score = score
                            best_threshold = thresh
                except Exception:
                    continue

        # 保存优化结果
        self.best_cutoff = best_threshold
        if threshold_metrics:
            self.threshold_metrics = pd.DataFrame(threshold_metrics)
            best_result = max(threshold_metrics, key=lambda x: x['cost_score'])
            print(f"最佳阈值: {best_threshold:.3f}")
            print(f"对应性能: 敏感度 {best_result['sensitivity']:.3f}, "
                  f"特异度 {best_result['specificity']:.3f}")
        else:
            print(f"使用默认阈值: {best_threshold:.3f}")

        return best_threshold

    def develop_decision_rules(self, df, X_test, y_test):

        print("构建女胎异常多层次判定规则...")

        # 检查模型可用性
        if len(self.trained_models) == 0:
            print("无法创建决策规则：模型不可用")
            return {}

        # 选择最佳模型
        best_model_name = self.evaluation_outcomes.loc[
            self.evaluation_outcomes['auc'].idxmax(), 'model'
        ]
        best_model = self.trained_models[best_model_name]

        # 获取预测概率
        probabilities = best_model.predict_proba(X_test)[:, 1]

        # 多层次规则框架 - 改进为5个级别
        risk_levels = {
            'direct_abnormal': [],  # 直接判为异常
            'high_risk': [],  # 高风险
            'medium_risk_retest': [],  # 中风险需复检
            'low_risk_observe': [],  # 低风险需观察
            'normal': []  # 正常
        }

        # 获取测试样本索引
        test_indices = X_test.index if hasattr(X_test, 'index') else range(len(X_test))
        test_features = self.construct_female_features(df.iloc[test_indices])

        # 应用多层次规则
        for i, (idx, row) in enumerate(test_features.iterrows()):
            # 计算最大Z值绝对值
            z_max_abs = max(abs(row['Z13']), abs(row['Z18']), abs(row['Z21']))
            # 获取预测概率
            prob = probabilities[i] if i < len(probabilities) else 0.5

            # 多层次判定逻辑
            if z_max_abs >= 3.5:
                # 1. 直接异常：Z值极端异常
                risk_levels['direct_abnormal'].append(i)
            elif prob >= 0.7:
                # 2. 高风险：预测概率很高
                risk_levels['high_risk'].append(i)
            elif 2.5 <= z_max_abs < 3.5 or (0.4 <= prob < 0.7):
                # 3. 中风险需复检：Z值中等异常或预测概率中等
                risk_levels['medium_risk_retest'].append(i)
            elif 0.2 <= prob < 0.4:
                # 4. 低风险需观察：预测概率低但不是很低
                risk_levels['low_risk_observe'].append(i)
            else:
                # 5. 正常：其他情况
                risk_levels['normal'].append(i)

        # 评估规则效果
        rule_metrics = {}
        for category, indices in risk_levels.items():
            if len(indices) > 0:
                if hasattr(y_test, 'iloc'):
                    actual_positives = y_test.iloc[indices].sum()
                else:
                    actual_positives = sum(y_test[indices])

                rule_metrics[category] = {
                    'count': len(indices),
                    'abnormal_count': actual_positives,
                    'precision': actual_positives / len(indices) if len(indices) > 0 else 0
                }

        # 输出规则统计
        print("风险分级统计:")
        for category, stats in rule_metrics.items():
            print(f"{category}: {stats['count']}例, 阳性率 {stats['precision']:.3f}")

        # 保存决策规则结果
        self.decision_framework = risk_levels
        self.rule_metrics = rule_metrics

        return risk_levels

    def evaluate_feature_influence(self):
        """分析特征重要性"""
        influence_data = []

        # 提取随机森林特征重要性
        if 'random_forest' in self.trained_models:
            rf_model = self.trained_models['random_forest']
            rf_importance = pd.DataFrame({
                'feature': self.feat_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            influence_data.append(('random_forest', rf_importance))

        # 提取L1逻辑回归系数
        if 'logistic_l1' in self.trained_models:
            lr_model = self.trained_models['logistic_l1']
            lr_coeffs = pd.DataFrame({
                'feature': self.feat_names,
                'coefficient': lr_model.coef_[0],
                'abs_coefficient': np.abs(lr_model.coef_[0])
            }).sort_values('abs_coefficient', ascending=False)
            influence_data.append(('logistic_l1', lr_coeffs))

        # 输出特征重要性
        if influence_data:
            print("特征重要性排序:")
            for model_name, df in influence_data:
                print(f"\n{model_name} 前10特征:")
                if 'importance' in df.columns:
                    print(df[['feature', 'importance']].head(10).to_string(index=False))
                else:
                    print(df[['feature', 'abs_coefficient']].head(10).to_string(index=False))

        return influence_data

    def visualize_outcomes(self, df, X_test, y_test):


        # 创建画布
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle('女胎异常判定模型分析结果', fontsize=16, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为顶部标题预留空间

        # 检查数据和模型可用性
        if X_test is None or y_test is None or len(self.trained_models) == 0:
            axes[0, 0].text(0.5, 0.5, '模型或数据不可用，无法生成结果', ha='center', va='center')
            plt.tight_layout()
            return fig

        # 获取最佳模型预测
        best_model_name = self.evaluation_outcomes.loc[self.evaluation_outcomes['auc'].idxmax(), 'model']
        best_model = self.trained_models[best_model_name]
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # --- 图1: ROC曲线 ---
        if len(np.unique(y_test)) > 1:
            models_to_plot = ['logistic_l1', 'random_forest']
            for name in models_to_plot:
                if name in self.trained_models:  # 确保模型存在
                    model = self.trained_models[name]
                    model_probs = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, model_probs)
                    auc = roc_auc_score(y_test, model_probs)
                    axes[0, 0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')

            # 添加基准线和装饰
            axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0, 0].set_xlabel('假阳性率')
            axes[0, 0].set_ylabel('真阳性率')
            axes[0, 0].set_title('ROC曲线分析')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, '样本类别单一，无法绘制ROC曲线', ha='center', va='center')

        # --- 图2: 精确率-召回率曲线 ---
        if len(np.unique(y_test)) > 1:
            models_to_plot = ['logistic_l1', 'random_forest']
            for name in models_to_plot:
                if name in self.trained_models:  # 确保模型存在
                    model = self.trained_models[name]
                    model_probs = model.predict_proba(X_test)[:, 1]
                    precision, recall, _ = precision_recall_curve(y_test, model_probs)
                    axes[0, 1].plot(recall, precision, label=name)
            # 添加装饰
            axes[0, 1].set_xlabel('召回率')
            axes[0, 1].set_ylabel('精确率')
            axes[0, 1].set_title('精确率-召回率曲线')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, '样本类别单一，无法绘制PR曲线', ha='center', va='center')

        # --- 图3: 校准曲线 ---
        if len(np.unique(y_test)) > 1 and len(y_prob) > 10:
            from sklearn.calibration import calibration_curve
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_test, y_prob, n_bins=min(5, len(y_test) // 2))
                axes[0, 2].plot(mean_predicted_value, fraction_of_positives, "s-", label=best_model_name)
                axes[0, 2].plot([0, 1], [0, 1], "k:", label="理想校准")
                axes[0, 2].set_xlabel('预测概率')
                axes[0, 2].set_ylabel('观测频率')
                axes[0, 2].set_title('概率校准曲线')
                axes[0, 2].legend()
            except:
                axes[0, 2].text(0.5, 0.5, '校准曲线生成失败', ha='center', va='center')
        else:
            axes[0, 2].text(0.5, 0.5, '样本量不足，无法绘制校准曲线', ha='center', va='center')

        # --- 图4: 随机森林特征重要性 ---
        if 'random_forest' in self.trained_models:
            rf = self.trained_models['random_forest']
            importances = rf.feature_importances_
            # 选取重要性最高的特征
            top_indices = np.argsort(importances)[-min(15, len(importances)):]

            axes[1, 0].barh(range(len(top_indices)), importances[top_indices])
            axes[1, 0].set_yticks(range(len(top_indices)))
            axes[1, 0].set_yticklabels([self.feat_names[i] for i in top_indices])
            axes[1, 0].set_xlabel('重要性得分')
            axes[1, 0].set_title('特征重要性分析 (随机森林)')

        # --- 图5: Z值分布 ---
        features = self.construct_female_features(df)
        z_cols = ['Z13', 'Z18', 'Z21']
        for col in z_cols:
            if col in features.columns:
                axes[1, 1].hist(features[col], alpha=0.5, label=col, bins=20)

        # 添加传统阈值线
        axes[1, 1].axvline(3.0, color='red', linestyle='--', alpha=0.7, label='传统阳性阈值(+)')
        axes[1, 1].axvline(-3.0, color='red', linestyle='--', alpha=0.7, label='传统阳性阈值(-)')
        axes[1, 1].set_xlabel('Z值')
        axes[1, 1].set_ylabel('样本数量')
        axes[1, 1].set_title('染色体Z值分布')
        axes[1, 1].legend()

        # --- 图6: 阈值优化结果 ---
        if hasattr(self, 'threshold_metrics') and len(self.threshold_metrics) > 0:
            thresh_df = self.threshold_metrics
            axes[1, 2].plot(thresh_df['threshold'], thresh_df['sensitivity'], 'o-', label='敏感度')
            axes[1, 2].plot(thresh_df['threshold'], thresh_df['specificity'], 's-', label='特异度')
            axes[1, 2].axvline(self.best_cutoff, color='red', linestyle='--', label='最优阈值')
            axes[1, 2].set_xlabel('决策阈值')
            axes[1, 2].set_ylabel('性能指标')
            axes[1, 2].set_title('阈值优化分析')
            axes[1, 2].legend()
        else:
            axes[1, 2].text(0.5, 0.5, '阈值优化数据不可用', ha='center', va='center')

        # --- 图7: 混淆矩阵 ---
        y_pred = (y_prob >= self.best_cutoff).astype(int)
        if len(np.unique(y_test)) > 1 and len(np.unique(y_pred)) > 1:
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2, 0])
            axes[2, 0].set_xlabel('预测标签')
            axes[2, 0].set_ylabel('真实标签')
            axes[2, 0].set_title(f'混淆矩阵 (阈值={self.best_cutoff:.3f})')
        else:
            axes[2, 0].text(0.5, 0.5, '无法生成有效的混淆矩阵', ha='center', va='center')

        # --- 图8: 决策规则分布 ---
        if hasattr(self, 'rule_metrics') and self.rule_metrics:
            rule_names = list(self.rule_metrics.keys())
            rule_counts = [self.rule_metrics[name]['count'] for name in rule_names]

            axes[2, 1].bar(range(len(rule_names)), rule_counts, alpha=0.7)
            axes[2, 1].set_xticks(range(len(rule_names)))
            axes[2, 1].set_xticklabels([name.replace('_', '\n') for name in rule_names], rotation=0)
            axes[2, 1].set_ylabel('样本数量')
            axes[2, 1].set_title('风险分级样本分布')
        else:
            axes[2, 1].text(0.5, 0.5, '决策规则统计不可用', ha='center', va='center')

        # --- 图9: 模型性能对比 ---
        if hasattr(self, 'evaluation_outcomes') and len(self.evaluation_outcomes) > 0:
            models_to_display = ['logistic_l1', 'random_forest']

            filtered_results = self.evaluation_outcomes[self.evaluation_outcomes['model'].isin(models_to_display)]

            model_names = filtered_results['model']
            aucs = filtered_results['auc']

            bars = axes[2, 2].bar(range(len(model_names)), aucs, alpha=0.7)
            axes[2, 2].set_xticks(range(len(model_names)))
            axes[2, 2].set_xticklabels([name.replace('_', '\n') for name in model_names], rotation=0)
            axes[2, 2].set_ylabel('AUC')
            axes[2, 2].set_title('模型性能对比')
            axes[2, 2].set_ylim([0.4, 1.0])

            # 添加数值标签
            for bar, auc in zip(bars, aucs):
                axes[2, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                f'{auc:.3f}', ha='center', va='bottom')
        else:
            axes[2, 2].text(0.5, 0.5, '模型评估结果不可用', ha='center', va='center')

        plt.tight_layout()
        return fig

    def produce_prediction_summary(self, df, X_test, y_test):

        if X_test is None or y_test is None or len(self.trained_models) == 0:
            print("无法生成报告：模型或数据不可用")
            return {}, {}

        # 获取最佳模型
        best_model_name = self.evaluation_outcomes.loc[self.evaluation_outcomes['auc'].idxmax(), 'model']
        best_model = self.trained_models[best_model_name]

        # 获取预测
        y_prob = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= self.best_cutoff).astype(int)

        # 生成分类报告
        if len(np.unique(y_test)) > 1 and len(np.unique(y_pred)) > 1:
            try:
                report = classification_report(y_test, y_pred, output_dict=True)
            except:
                report = {}
        else:
            report = {}

        # 高风险样本分析
        high_risk_mask = y_prob >= 0.8
        high_risk_indices = np.where(high_risk_mask)[0]
        high_risk_actual = y_test[high_risk_indices] if len(high_risk_indices) > 0 else []

        # 生成摘要
        overview = {
            'model_used': best_model_name,
            'optimal_threshold': self.best_cutoff,
            'total_samples': len(y_test),
            'predicted_abnormal': int(sum(y_pred)),
            'actual_abnormal': int(sum(y_test)),
            'high_risk_samples': len(high_risk_indices),
            'high_risk_accuracy': sum(high_risk_actual) / len(high_risk_indices) if len(high_risk_indices) > 0 else 0,
            'overall_accuracy': (y_pred == y_test).mean() if len(y_test) > 0 else 0,
            'auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
        }

        # 添加分类指标
        if report and '1' in report:
            overview.update({
                'precision': report['1'].get('precision', 0),
                'recall': report['1'].get('recall', 0),
                'f1_score': report['1'].get('f1-score', 0)
            })
        else:
            overview.update({
                'precision': 0,
                'recall': 0,
                'f1_score': 0
            })

        return overview, report



def execute_problem4():

    print("=" * 60)
    print("问题4：女胎异常判定分析")
    print("=" * 60)

    # 初始化数据处理器
    processor = NIPTDataProcessor()

    # 加载数据
    if not processor.load_dataset('女胎.csv'):
        print("数据加载失败，请检查女胎.csv文件")
        return None

    # 预处理女胎数据
    female_data = processor.preprocess_female_records()
    if female_data is None or len(female_data) == 0:
        print("女胎数据预处理失败")
        return None

    print(f"女胎数据统计: {len(female_data)}例")
    print(f"异常样本统计: {sum(~female_data['AB'].isna())}例")

    # 创建求解器
    solver = Problem4Solver(processor)

    # 第一步：训练模型
    print("\n第一步：训练女胎异常判定模型")
    X_test, y_test = solver.build_models(female_data)

    if X_test is not None and y_test is not None:
        # 第二步：优化决策阈值
        print("\n第二步：优化决策阈值")
        optimal_threshold = solver.adjust_decision_threshold(X_test, y_test)

        # 第三步：创建多层次决策规则
        print("\n第三步：创建多层次决策规则")
        decision_rules = solver.develop_decision_rules(female_data, X_test, y_test)

        # 第四步：分析特征重要性
        print("\n第四步：分析特征重要性")
        importance_results = solver.evaluate_feature_influence()

        # 第五步：生成预测报告
        print("\n第五步：生成预测分析报告")
        overview, detailed_report = solver.produce_prediction_summary(female_data, X_test, y_test)

        # 第六步：绘制可视化结果
        print("\n第六步：生成可视化结果")
        fig = solver.visualize_outcomes(female_data, X_test, y_test)
        plt.savefig('Q4results/problem4_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 第七步：保存分析结果
        print("\n第七步：保存分析结果")


        print("=" * 60)
        print("问题4分析完成！")
        print("生成的文件：")
        print("- Q4results/problem4_analysis.png (可视化结果)")
        print("=" * 60)

        # 输出关键结果摘要
        print("\n=== 关键结果摘要 ===")
        for key, value in overview.items():
            if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
            else:
                    print(f"{key}: {value}")

            # 决策规则建议
            print("\n=== 女胎异常判定建议 ===")
            print("1. 直接异常判定: Z值最大绝对值 ≥ 3.5")
            print("2. 高风险: 模型预测概率 ≥ 0.7")
            print("3. 中风险复检: 2.5 ≤ Z值最大绝对值 < 3.5 或 0.4 ≤ 模型预测概率 < 0.7")
            print("4. 低风险观察: 0.2 ≤ 模型预测概率 < 0.4")
            print("5. 正常: 其他情况")

            print(f"\n最佳模型: {overview['model_used']}")
            print(f"整体准确率: {overview['overall_accuracy']:.3f}")
            print(f"AUC: {overview['auc']:.3f}")

            if hasattr(solver, 'rule_metrics'):
                print("\n各风险级别性能:")
                for rule_name, stats in solver.rule_metrics.items():
                    print(f"- {rule_name}: {stats['count']}例, 阳性率 {stats['precision']:.3f}")

        else:
            print("没有设置生成预测报告")

    else:
        print("警告：模型训练失败，无法执行后续分析")

    return solver



if __name__ == "__main__":
    execute_problem4()