import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt

# タイトルとサブタイトルの表示
st.title("決定木分析ツール")
st.markdown("### データ分析と意思決定を支援")
st.markdown("**Decision Tree Analysis Tool** *| Supporting Data Analysis and Decision-Making*")

# CSVファイルのアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

# 決定木の最大深さスライダー
max_depth = st.slider("決定木の最大深さ", min_value=1, max_value=200, value=5, step=1)

if uploaded_file is not None:
    # CSVファイルの読み込み
    data = pd.read_csv(uploaded_file)
    
    # 欠損値がある行を削除（必要に応じてfillnaなどで補完してもよい）
    data.dropna(inplace=True)
    
    # 文字列やカテゴリ型の列を数値に変換
    for col in data.columns:
        if data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data[col]):
            data[col] = data[col].astype('category').cat.codes
    
    # 目的変数y（データの最左列）と説明変数X（残り）
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]
    
    # 学習データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 決定木モデルの作成と学習
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # テストデータで予測し、精度を表示
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"モデルの精度: {accuracy:.2f}")
    
    # 特徴量重要度の表示
    st.write("特徴量の重要度:")
    feature_importances = pd.DataFrame({
        '特徴量': X.columns,
        '重要度': model.feature_importances_
    }).sort_values(by='重要度', ascending=False)
    st.write(feature_importances)
    
    # 決定木の樹形図を表示
    st.write("決定木の樹形図:")
    num_nodes = model.tree_.node_count
    fig_width = min(24, 10 + num_nodes // 8)
    fig_height = min(18, 8 + num_nodes // 12)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_tree(model, feature_names=X.columns, filled=True, ax=ax, fontsize=10, proportion=False)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
    st.pyplot(fig)

else:
    st.write("CSVファイルを上の枠内にドラッグ＆ドロップすれば分析が始まります。")
