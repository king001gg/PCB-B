"""喷砂参数—表面质量—阻焊附着力关联分析脚本。

对采集的数据集进行统计分析，建立喷砂参数与阻焊附着力之间的回归模型，
为企业提供工艺窗口优化建议。

输入数据格式（CSV）：
    pressure_mpa, grit_mesh, feed_speed_mps, roughness_cv_mean,
    oxidation_pct, embedding_count, unroughened_pct,
    overall_score, adhesion_force_N

用法：
    python scripts/process_analysis.py --data process_data.csv
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_process_data(csv_path: str) -> dict:
    """加载工艺参数数据。

    返回包含各列数组的字典。
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"加载 {len(df)} 条记录，列: {list(df.columns)}")
        return df
    except ImportError:
        print("[WARN] pandas 未安装，使用纯 Python 解析（功能受限）")
        print("  安装: pip install pandas")
        # 简单 CSV 解析回退
        data = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            for col in header:
                data[col.strip()] = []
            for line in f:
                values = line.strip().split(",")
                for i, v in enumerate(values):
                    col = header[i].strip()
                    try:
                        data[col].append(float(v))
                    except ValueError:
                        data[col].append(v)
        return data


def analyze_correlation(df) -> dict:
    """计算各参数与附着力的皮尔逊相关系数。"""
    target = "adhesion_force_N"
    if target not in df.columns:
        print(f"[WARN] 数据中未找到目标列 '{target}'，跳过相关性分析")
        return {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = {}
    for col in numeric_cols:
        if col == target:
            continue
        corr = df[col].corr(df[target])
        correlations[col] = round(corr, 4)

    # 排序
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]),
                          reverse=True)
    print("\n=== 与拉拔力的相关系数（按 |r| 降序） ===")
    for col, r in sorted_corr:
        strength = "强" if abs(r) > 0.7 else ("中" if abs(r) > 0.4 else "弱")
        direction = "正" if r > 0 else "负"
        print(f"  {col:30s}: r={r:+.4f} ({strength}{direction}相关)")
    return correlations


def fit_regression_model(df) -> dict:
    """拟合多元线性回归模型，预测阻焊附着力。"""
    target = "adhesion_force_N"
    if target not in df.columns:
        return {}

    feature_cols = [
        "roughness_cv_mean", "oxidation_pct",
        "embedding_count", "unroughened_pct",
        "overall_score",
    ]
    # 过滤存在的特征列
    features = [c for c in feature_cols if c in df.columns]

    X = df[features].values
    y = df[target].values

    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import r2_score, mean_absolute_error
    except ImportError:
        print("[WARN] scikit-learn 未安装，跳过回归分析")
        print("  安装: pip install scikit-learn")
        return {}

    # 简单线性回归（小数据集直接用全部数据）
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print("\n=== 阻焊附着力回归模型 ===")
    print(f"  R² = {r2:.4f}")
    print(f"  MAE = {mae:.2f} N")
    print("  系数:")
    for name, coef in zip(features, model.coef_):
        print(f"    {name:30s}: {coef:+.4f}")
    print(f"    截距{'':28s}: {model.intercept_:+.2f}")

    return {
        "r2": r2,
        "mae": mae,
        "coefficients": dict(zip(features, model.coef_)),
        "intercept": model.intercept_,
    }


def process_window_optimization(df) -> dict:
    """工艺窗口优化建议。

    基于 visual score（整体评分）和对各参数的回归，
    推荐能获得最高质量的喷砂参数组合。
    """
    print("\n=== 工艺窗口优化建议 ===")

    suggestions = []

    # 分析各压力水平下的平均评分
    if "pressure_mpa" in df.columns and "overall_score" in df.columns:
        for pressure in sorted(df["pressure_mpa"].unique()):
            subset = df[df["pressure_mpa"] == pressure]
            avg_score = subset["overall_score"].mean()
            avg_adhesion = subset.get("adhesion_force_N")
            avg_adhesion = avg_adhesion.mean() if avg_adhesion is not None else None

            print(f"  压力 {pressure} MPa: "
                  f"均分={avg_score:.1f}, "
                  f"平均附着力={avg_adhesion:.1f} N" if avg_adhesion else
                  f"均分={avg_score:.1f}")

            if avg_score >= 80:
                suggestions.append(
                    f"推荐压力 {pressure} MPa（均分 {avg_score:.1f} ≥ 80）"
                )

    # 分析磨料目数
    if "grit_mesh" in df.columns:
        for grit in sorted(df["grit_mesh"].unique()):
            subset = df[df["grit_mesh"] == grit]
            avg_score = subset["overall_score"].mean()
            print(f"  磨料 {grit} 目: 均分={avg_score:.1f}")

            if avg_score >= 80:
                suggestions.append(
                    f"推荐磨料 {grit} 目（均分 {avg_score:.1f} ≥ 80）"
                )

    if suggestions:
        print("\n推荐工艺窗口:")
        for s in suggestions:
            print(f"  ✓ {s}")
    else:
        print("  数据不足，无法提供具体建议。请采集更多数据。")

    return {"suggestions": suggestions}


def main():
    parser = argparse.ArgumentParser(
        description="喷砂工艺参数—质量—附着力关联分析"
    )
    parser.add_argument(
        "--data", required=True,
        help="工艺参数 CSV 数据文件"
    )
    parser.add_argument(
        "--generate-sample", action="store_true",
        help="生成示例数据文件"
    )
    args = parser.parse_args()

    if args.generate_sample:
        # 生成示例数据
        sample_csv = """pressure_mpa,grit_mesh,feed_speed_mps,roughness_cv_mean,oxidation_pct,embedding_count,unroughened_pct,overall_score,adhesion_force_N
0.4,180,0.05,0.18,2.1,5,1.2,82.5,45.2
0.4,180,0.05,0.22,3.5,8,1.8,75.3,42.1
0.4,180,0.08,0.25,2.8,6,2.5,71.0,38.5
0.4,220,0.05,0.15,1.8,3,0.8,88.2,48.9
0.4,220,0.05,0.16,2.0,4,1.0,86.7,47.3
0.4,220,0.08,0.20,2.3,5,1.5,79.1,44.0
0.6,180,0.05,0.12,1.5,4,0.5,90.1,52.3
0.6,180,0.05,0.14,1.8,5,0.7,87.5,50.8
0.6,180,0.08,0.17,2.0,6,1.1,83.0,48.1
0.6,220,0.05,0.10,1.2,2,0.3,93.4,56.7
0.6,220,0.05,0.11,1.3,3,0.4,92.0,55.2
0.6,220,0.08,0.14,1.6,4,0.6,88.5,51.9
0.8,180,0.05,0.15,2.5,7,1.0,84.2,49.0
0.8,180,0.05,0.18,3.0,9,1.3,79.8,46.3
0.8,180,0.08,0.21,3.8,11,1.8,73.5,42.8
0.8,220,0.05,0.13,2.0,5,0.8,86.1,50.1
0.8,220,0.05,0.14,2.2,6,0.9,84.7,48.9
0.8,220,0.08,0.18,2.8,8,1.4,78.2,45.5
"""
        with open(args.data, "w", encoding="utf-8") as f:
            f.write(sample_csv)
        print(f"示例数据已生成: {args.data}")
        print("运行分析: python scripts/process_analysis.py --data " + args.data)
        return

    # 加载数据
    df = load_process_data(args.data)

    # 相关性分析
    correlations = analyze_correlation(df)

    # 回归模型
    regression = fit_regression_model(df)

    # 工艺窗口优化
    optimization = process_window_optimization(df)

    # 汇总
    print("\n" + "=" * 50)
    print("分析完成。建议:")
    if "overall_score" in df.columns:
        best = df.loc[df["overall_score"].idxmax()]
        print(f"  最佳参数组合: "
              f"压力={best.get('pressure_mpa', '?')} MPa, "
              f"磨料={best.get('grit_mesh', '?')} 目, "
              f"传送速度={best.get('feed_speed_mps', '?')} m/s")
        print(f"  对应质量: 评分={best['overall_score']:.1f}")
        if "adhesion_force_N" in best:
            print(f"  对应附着力: {best['adhesion_force_N']:.1f} N")


if __name__ == "__main__":
    main()
