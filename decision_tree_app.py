import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt

st.title("決定木アプリ（Decision Tree Application）")
st.markdown("### 作成者: 土居拓務（DOI, Takumu）")

# 利用上の注意や目的を表示
st.markdown("""
本アプリを使用して分析する際は、以下の点にご留意ください。

1. **分析の目的**：本アプリは教育や研究用途での利用を目的としています。
2. **データの適切な前処理**：精度向上のため、欠損値や異常値を含むデータは適切に処理してください。
3. **結果の解釈**：本アプリの結果はデータに基づくものであり、必ずしも全てのケースにおいて最適な判断が得られるわけではありません。
""")

# 使用法の説明を追加
st.markdown("""
### アプリの使用法

このアプリは、CSVファイルを分析するためのツールです。使用方法は以下の通りです。

1. **CSVファイルの準備**:
   - 最左列には目的変数を記載してください。
   - 残りの列は説明変数として扱われます。

2. **列名の設定**:
   - 各変数の名称は最上列に記載されます。この列には、変数名を短めの半角英数字で記入することをお勧めします。

3. **分析の実行**:
   - CSVファイルをアップロードすると、自動的に目的変数と説明変数を分析します。

この手順に従ってCSVファイ
