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


