# XGBoost-Feature-Name

How to access feature names in a trained XGBoost model?

vocus link: https://vocus.cc/article/64d32f43fd8978000115822f


## English Version

### The Story Behind This
When I took over a project, I received a trained pickle file containing the weights of an XGBoost model.  
Because the feature engineering code had already been handed over multiple times, and I was in a hurry to run inference on new data, I made a small mistake that turned into a costly one: I did not check the model's feature names first.

This article is a reminder to my future self not to repeat that mistake.

### Goal
This article shows how to retrieve the feature names from a trained XGBoost model and includes sample code based on the Kaggle Titanic dataset.

### Train an XGBoost Model
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('/kaggle/input/titanic/train.csv')

# Select features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

X = data[features]
y = data[target]

# Handle missing values and categorical features
X['Age'].fillna(X['Age'].median(), inplace=True)
X['Embarked'].fillna(X['Embarked'].mode()[0], inplace=True)
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X['Embarked'] = X['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# One-hot encode categorical features
X = pd.get_dummies(X, columns=['Embarked'], drop_first=True)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost classifier
model = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=3)

# Train model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

Expected output:

```python
Accuracy: 0.8212290502793296
```

### Save the Weights as a Pickle File
```python
import pickle

with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(model, file)
```

### Load the Pickle File
```python
import pickle

# Load model from pickle file
with open('xgboost_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
```

### The Main Point: Check Feature Names from the Pickle File
In XGBoost, a model consists of many trees, also called boosters, and they expose useful metadata about the trained model.  
We can call `get_booster()` on the loaded model and inspect the `feature_names` attribute to see which features were actually used.

```python
clf = loaded_model.get_booster()
print(clf.feature_names)
print(type(clf))
```

Output:

```python
['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_1', 'Embarked_2']
<class 'xgboost.core.Booster'>
```

Now we can clearly see which feature names the model expects, as well as the input order of those features.

That is the key takeaway. I hope this helps anyone trying to recover feature names from a trained XGBoost model.


## 中文版

### 故事是這樣的...
在接手某個專案時，我拿到了一份已經訓練好的 pickle 檔案，裡面記錄著 XGBoost model weight。  
但因為 feature engineering 的程式碼已經交接了幾手，而我又急著拿這個模型去對新資料做 inference，就犯下了一個看似微不足道、卻讓後續工作全部白做的錯誤：沒有先檢查 model 的 feature name。  

所以才有了這篇文章，提醒未來的自己不要再忘記這次的經驗。

### 今天的目標
這篇文章會記錄如何透過 XGBoost 提供的方法，取得已訓練模型中的 feature names，並附上 sample code（以 Kaggle Titanic dataset 為例）。

### 訓練一個 XGBoost model
```python
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

正確輸出：

```python
Accuracy: 0.8212290502793296
```

### 儲存 weight 成為 pickle 檔案
```python
import pickle

with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(model, file)
```

### 讀取 pickle
```python
import pickle

# 從 Pickle 文件中載入模型
with open('xgboost_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
```

### 今天的重頭戲：通過 pickle 確認 feature name
在 XGBoost 中，模型由許多樹（boosters）組成，每棵樹都能提供模型的部分資訊。  
我們可以使用 `get_booster()` 方法，從載入後的模型中取得 `Booster` 物件，再透過 `feature_names` 屬性，就能知道模型實際使用了哪些特徵名稱。

```python
clf = loaded_model.get_booster()
print(clf.feature_names)
print(type(clf))
```

會得到輸出：

```python
['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_1', 'Embarked_2']
<class 'xgboost.core.Booster'>
```

這樣就能確認模型實際使用到哪些特徵，以及 feature 輸入的順序。

好啦，終於完成了。希望有幫助到正在找 feature name 的人，我們下次見。

重要的時刻常常出現在微小縫隙，所以要保持警覺，因為這些時刻可能改變一切。


