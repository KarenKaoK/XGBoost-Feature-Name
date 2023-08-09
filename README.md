# XGBoost-Feature-Name

- vocus link : https://vocus.cc/article/64d32f43fd8978000115822f

How to access feature names in a trained XGB model ?

## 故事是這樣的...
在接手某個專案中，取得了一份已經訓練好的 pickle 檔案記載著 XGBoost model weight ，但因為 feature engineering 的程式碼交接了幾手，而我急於使用這個模型來 inference 新的數據，就在此刻犯下了一個看似微不足道但導致後續作業都成了白工的錯誤:沒有檢查 model 的 feature name 
所以就有這篇文章～ 提醒未來的自己不要忘記這次的經驗！那開始今天的分享～

## 故事說完了，來說今天的目標
這篇文章將紀錄如何通過 XGBoost 提供的方法來獲取已訓練模型中的特徵名稱，並附上 sample code (以 Kaggle titanic dataset 為例)。

## 訓練一個 XGBoost model 
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 讀取數據集
data = pd.read_csv('/kaggle/input/titanic/train.csv')

# 選擇特徵和目標變量
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

X = data[features]
y = data[target]

# 處理缺失值和類別特徵
X['Age'].fillna(X['Age'].median(), inplace=True)
X['Embarked'].fillna(X['Embarked'].mode()[0], inplace=True)
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X['Embarked'] = X['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# 將類別特徵轉換成獨熱編碼
X = pd.get_dummies(X, columns=['Embarked'], drop_first=True)

# 將數據集分成訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化 XGBoost 分類器
model = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=3)

# 訓練模型
model.fit(X_train, y_train)

# 預測測試集
y_pred = model.predict(X_test)

# 計算準確率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

正確輸出 : Accuracy: 0.8212290502793296

## 儲存 weight 成為 pickle 檔案

```
import pickle
with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(model, file)
```    

## 讀取 pickle 
```

import pickle

# 從 Pickle 文件中載入模型
with open('xgboost_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
```

## 今天的重頭戲: 通過 pickle 確認 feature name 

在 XGBoost 中，模型由許多樹（boosters）組成，每個樹都可以提供有關模型的一些信息。我們可以使用 get_booster() 方法從載入的模型中獲取 Booster 物件，然後通過查詢 feature_names 屬性，我們可以獲得模型中使用的特徵名稱。

```
clf = loaded_model.get_booster()
print(clf.feature_names)   
print(type(clf))
```

會得到輸出:

```
['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_1', 'Embarked_2']
<class 'xgboost.core.Booster'>
```

就可以知道這次使用到的特徵名稱為哪些，和 feature 輸入的順序！

好啦～終於完成了！希望有幫助到在找 feature_name 的人，我們下次見！

重要的時刻常常出現在微小縫隙，所以要保持警覺，因為這些時刻可能改變一切。

