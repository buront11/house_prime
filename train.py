import pandas as pd # 基本ライブラリ
import numpy as np # 基本ライブラリ
import matplotlib.pyplot as plt # グラフ描画用
import seaborn as sns; sns.set() # グラフ描画用
import warnings # 実行に関係ない警告を無視
warnings.filterwarnings('ignore')
import lightgbm as lgb #LightGBM
from sklearn import datasets
from sklearn.model_selection import train_test_split # データセット分割用
from sklearn.metrics import mean_squared_error # モデル評価用(平均二乗誤差)
from sklearn.metrics import r2_score # モデル評価用(決定係数)

def main():
    df = pd.read_csv('./preprocessed_train.csv')

    X = df.drop('SalePrice', axis=1).values
    y = df['SalePrice'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=2)

    model = lgb.LGBMRegressor() # モデルのインスタンスの作成
    model.fit(X_train, y_train) # モデルの学習

    # テストデータの予測
    y_pred = model.predict(X_test)

if __name__=='__main__':
    main()