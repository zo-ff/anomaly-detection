import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# =====================================================================
# 1. 時系列ダミーデータの作成
# =====================================================================
print("--- 1. 時系列ダミーデータの作成 ---")
np.random.seed(42)

# 2026年1月1日〜2026年3月31日までの日次データを作成（約90日分）
dates = pd.date_range(start="2026-01-01", end="2026-03-31", freq="D")
n_samples = len(dates)

# トレンド + 曜日周期（7日サイクル） + ランダムノイズ で売上データをシミュレート
trend = np.linspace(100, 150, n_samples)
weekly_cycle = np.sin(2 * np.pi * dates.dayofweek / 7) * 20
noise = np.random.normal(0, 5, n_samples)
sales = trend + weekly_cycle + noise

df = pd.DataFrame({"sales": sales}, index=dates)
print(f"データ件数: {len(df)}件 (期間: {df.index.min().date()} 〜 {df.index.max().date()})")

# =====================================================================
# 2. 特徴量エンジニアリング（時系列特有の特徴量）
# =====================================================================
print("\n--- 2. 特徴量エンジニアリング（ラグ & 移動平均 & カレンダー） ---")

# ターゲット（予測したい値）
target_col = "sales"

# 【ラグ特徴量 (Lag Features)】
# LightGBMに「過去のパターン」を教えるために、過去の値をずらして列として与えます。
df["lag_1"] = df[target_col].shift(1)  # 1日前
df["lag_7"] = df[target_col].shift(7)  # 7日前

# 【ローリング特徴量 (Rolling Features)】
# 過去3日間の平均値（リークを防ぐため、当日は含めず1日前から計算します）
df["rolling_mean_3"] = df[target_col].shift(1).rolling(window=3).mean()

# 【カレンダー特徴量 (Calendar Features)】
# 曜日や日付そのものも重要な手がかりになります。
df["dayofweek"] = df.index.dayofweek
df["day"] = df.index.day

# NaN（ラグによって生じる過去データがない行）を削除
df = df.dropna()

# 特徴量(X)とターゲット(y)に分離
feature_cols = ["lag_1", "lag_7", "rolling_mean_3", "dayofweek", "day"]
X = df[feature_cols]
y = df[target_col]

print(f"作成した特徴量: {feature_cols}")
print(df.head(5))

# =====================================================================
# 3. 時系列の順序に沿ったデータの分割
# =====================================================================
print("\n--- 3. 時系列データの分割 ---")
# 時系列データはランダムに分割してはいけません！「過去」で学習し「未来」を予測します。
# ここでは最後の15日間を検証（Validation）データにします。
split_date = df.index[-15]

X_train = X.loc[:split_date]
y_train = y.loc[:split_date]
X_val = X.loc[split_date:]
y_val = y.loc[split_date:]

print(f"学習データ期間: {X_train.index.min().date()} 〜 {X_train.index.max().date()} ({len(X_train)}件)")
print(f"検証データ期間: {X_val.index.min().date()} 〜 {X_val.index.max().date()} ({len(X_val)}件)")

# =====================================================================
# 4. Native API での学習と予測（おすすめ・カスタマイズしやすい）
# =====================================================================
print("\n--- 4. Native API での学習開始 ---")

# (A) LightGBM専用の Dataset オブジェクトを作成する
train_dataset = lgb.Dataset(X_train, label=y_train)
val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

# (B) パラメータを辞書型で設定する
params = {
    "objective": "regression",      # 回帰問題
    "metric": "rmse",               # 評価指標：RMSE
    "learning_rate": 0.05,          # 学習率
    "num_leaves": 15,               # 葉の最大数（小さめのデータなので抑えめに）
    "seed": 42,
    "verbose": -1
}

# (C) 早期停止 (Early Stopping) などの制御用コールバック
callbacks = [
    lgb.early_stopping(stopping_rounds=10, verbose=True),
    lgb.log_evaluation(period=5)
]

# (D) 学習を実行する (lgb.train)
bst = lgb.train(
    params,
    train_dataset,
    num_boost_round=100,            # 最大イテレーション数
    valid_sets=[train_dataset, val_dataset],
    callbacks=callbacks
)

# (E) 予測する
y_pred_native = bst.predict(X_val, num_iteration=bst.best_iteration)
rmse_native = np.sqrt(mean_squared_error(y_val, y_pred_native))
print(f"Native API モデルの検証データ RMSE: {rmse_native:.4f}")

# =====================================================================
# 5. scikit-learn API での学習と予測（シンプル）
# =====================================================================
print("\n--- 5. scikit-learn API での学習開始 ---")

# (A) モデルのインスタンス化（ハイパーパラメータを引数で渡す）
model = lgb.LGBMRegressor(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    n_estimators=100,               # Native APIの num_boost_round に相当
    num_leaves=15,
    random_state=42,
    verbose=-1
)

# (B) 学習を実行する (fit)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=10, verbose=True),
        lgb.log_evaluation(period=5)
    ]
)

# (C) 予測する
y_pred_sklearn = model.predict(X_val)
rmse_sklearn = np.sqrt(mean_squared_error(y_val, y_pred_sklearn))
print(f"scikit-learn API モデルの検証データ RMSE: {rmse_sklearn:.4f}")

# =====================================================================
# 6. 結果の可視化（プロット）
# =====================================================================
print("\n--- 6. 予測結果の可視化 ---")
plt.figure(figsize=(10, 5))
plt.plot(y_train.index, y_train, label="Train (Actual)", color="blue", alpha=0.6)
plt.plot(y_val.index, y_val, label="Validation (Actual)", color="green", marker='o')
plt.plot(y_val.index, y_pred_native, label="Predicted (Native API)", color="red", linestyle="--", marker='x')
plt.title("LightGBM Sales Forecasting (Time Series Practice)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

output_img = "lightgbm_forecast_result.png"
plt.savefig(output_img, dpi=150)
print(f"🎉 予測結果のグラフを '{output_img}' に保存しました。")
