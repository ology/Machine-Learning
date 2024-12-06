import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import re
import string

df1 = pd.read_csv('titanic-1.csv')
# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# UNIQUE: SibSp,Parch,Fare
df1.dropna(inplace=True, subset=['Survived'])
df1['sanitized'] = df1['Name'].replace(r'[' + string.punctuation + r']', '', regex=True)
df1 = df1.sort_values(by='sanitized')
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df1['sanitized'])

df2 = pd.read_csv('titanic-2.csv')
# "row.names","pclass","survived","name","age","embarked","home.dest","room","ticket","boat","sex"
# UNIQUE: "home.dest","boat"
df2.dropna(inplace=True, subset=['survived'])
df2['sanitized'] = df2['name'].replace(r'[' + string.punctuation + r']', '', regex=True)
df2 = df2.sort_values(by='sanitized')
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df2['sanitized'])

df2.rename(columns={
    "row.names": 'PassengerId',
    "pclass": 'Pclass',
    "survived": 'Survived',
    "name": 'Name',
    "age": 'Age',
    "embarked": 'Embarked',
    "home.dest": 'Home',
    "room": 'Cabin',
    "ticket": 'Ticket',
    "boat": 'Boat',
    "sex": 'Sex',
}, inplace=True)

merged = []
for row1 in df1.itertuples():
    match1 = re.search(r'^(\w+ \w+( \w*)?)', row1.sanitized)
    found = 0
    for row2 in df2.itertuples():
        match2 = re.search(r'^(\w+ \w+( \w*)?)', row2.sanitized)
        if match1.group(1) == match2.group(1):
            found = 1
            row = pd.DataFrame([row2])
            row['SibSp'] = pd.NA
            row['Parch'] = pd.NA
            row['Fare'] = pd.NA
            d = row.to_dict('records')
            merged.append(d[0])
            break
    if not found:
        found = 0
        row = pd.DataFrame([row1])
        row['Boat'] = pd.NA
        row['Home'] = pd.NA
        d = row.to_dict('records')
        merged.append(d[0])

df3 = pd.DataFrame(merged)
df3.drop(columns=['Index', 'sanitized'], inplace=True)
df3['Pclass'] = df3['Pclass'].astype(str)
df3['Pclass'] = df3['Pclass'].str[0]
df3['Pclass'] = df3['Pclass'].astype(int)
df3.loc[df3['Embarked'] == 'C', 'Embarked'] = 'Cherbourg'
df3.loc[df3['Embarked'] == 'S', 'Embarked'] = 'Southampton'
df3.loc[df3['Embarked'] == 'Q', 'Embarked'] = 'Queenstown'
df3.loc[df3['Sex'] == 'male', 'Sex'] = 0
df3.loc[df3['Sex'] == 'female', 'Sex'] = 1
df3['SibSp'] = df3['SibSp'].fillna(0)
df3['Parch'] = df3['Parch'].fillna(0)
df3['Fare'] = df3['Fare'].fillna(0)
df3['Age'] = df3['Age'].fillna(df3['Age'].mean())
df3['Embarked'] = df3['Embarked'].fillna('Unknown')
df3['Home'] = df3['Home'].fillna('Unknown')
df3['Cabin'] = df3['Cabin'].fillna('Unknown')
df3['Ticket'] = df3['Ticket'].fillna('Unknown')
df3['Boat'] = df3['Boat'].fillna('Unknown')
# print(df3.describe())#.info())
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df3['Embarked'])

df4 = df3.copy()
df4['Boat'] = df4['Boat'].astype('category').cat.codes
df4.loc[df4['Boat'] == -1, 'Boat'] = pd.NA
df4['Embarked'] = df4['Embarked'].astype('category').cat.codes
df4.loc[df4['Embarked'] == -1, 'Embarked'] = pd.NA
df4['Home'] = df4['Home'].astype('category').cat.codes
df4.loc[df4['Home'] == -1, 'Home'] = pd.NA
# print(df4.describe())#info())
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df4)

X = df4.drop(['Survived', 'Name', 'Home', 'Embarked', 'Ticket', 'Cabin', 'Boat'], axis=1)
y = df4['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
