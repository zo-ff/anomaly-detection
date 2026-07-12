# 決定木 学習メモ(Titanic演習)

## Titanicのデータ列の意味
- `PassengerId`: 単なる連番。予測には使わない。
- `Survived`: 生存フラグ(0=死亡, 1=生存)。今回の目的変数。
- `Pclass`: チケットの等級(1〜3)。等級が高いほど生存率に有利な傾向。
- `Name`: 氏名。そのままは使わないが、敬称(Mr./Miss.など)を抜き出して特徴量にする応用あり。
- `Sex`: 性別。
- `Age`: 年齢。欠損値が多い。
- `SibSp`: 同乗している兄弟姉妹(Sibling)+配偶者(Spouse)の人数。
- `Parch`: 同乗している親(Parent)+子供(Child)の人数。
- `Fare`: 運賃。
- `Cabin`: 客室番号。欠損が非常に多いので今回は不使用。
- `Embarked`: 乗船港(C=Cherbourg, Q=Queenstown, S=Southampton)。

## 特徴量エンジニアリングとは
生データをそのまま使うのではなく、モデルが学習しやすいように新しい列を作ったり加工したりすること。
例: `SibSp + Parch → FamilySize`(同乗家族の合計人数)、`Name → Title`(敬称を抽出)。

## 決定木 vs 線形回帰
- 線形回帰: 欠損値があると計算できない(補完が必須)。スケール(単位の大小)にも敏感。
- 決定木: しきい値で分岐するだけなのでスケール不要。sklearn 1.3以降はNaNも直接扱える。

## カテゴリ変数のエンコーディング
文字列(`male`/`female`など)はモデルに渡す前に数値化が必要。
```python
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X['Embarked'] = X['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
```
決定木は数字の大小に意味を持たせなくても分岐で対応できるので、one-hotでなく単純なmapでも実用上問題ない。

## train_test_split
学習データと検証データを分けて、未知データへの汎化性能を確認するために使う。
```python
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
```
- `test_size=0.2`: 20%を検証用に。
- `random_state=0`: 分割のランダム性を固定(再現性のため)。数字自体に意味はない。
- 命名(`X_train`, `X_valid`, `y_train`, `y_valid`)は業界の慣習であって予約語ではない。

## DecisionTreeClassifierの学習
```python
model = DecisionTreeClassifier(max_depth=3, random_state=0)
model.fit(X_train, y_train)
```
- `max_depth=3`: 木の深さの上限。浅く制限することで過学習を防ぎ、可視化もしやすくする。
- `random_state=0`: 同点の分岐候補があるときの選び方を固定(train_test_splitのrandom_stateとは別物の乱数制御)。

## 木の可視化
```python
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=['Died', 'Survived'], filled=True)
plt.show()
```
各分岐でどの特徴量・しきい値で分けたか、サンプル数、クラス内訳(gini係数)が図として確認できる。

## 環境まわりの小技
- Windows日本語環境でsklearnのモデルをそのまま表示するとHTML描画で`UnicodeDecodeError`(cp932)が出ることがある → `from sklearn import set_config; set_config(display='text')`で回避。
- 新しいPython環境でライブラリを初めてimportすると、Windows Defenderのスキャンで数十秒〜1分かかることがある(2回目以降は速い)。