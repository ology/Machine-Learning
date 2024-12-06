# from sklearn.datasets import fetch_openml
# X, y = fetch_openml("titanic", version=7, as_frame=True, return_X_y=True)
# print(X, y)

# Import libraries
import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score
import re
import string

df1 = pd.read_csv('titanic-1.csv')
# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# UNIQUE: SibSp,Parch,Fare
# print(df1.describe())
# print(df1.info())
df1.dropna(inplace=True, subset=['Survived','Pclass','Sex','Age'])
df1['sanitized'] = df1['Name'].replace(r'[' + string.punctuation + r']', '', regex=True)
df1 = df1.sort_values(by='sanitized')
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df1['sanitized'])
# print(df1)
# print(len(df1))

df2 = pd.read_csv('titanic-2.csv')
# "row.names","pclass","survived","name","age","embarked","home.dest","room","ticket","boat","sex"
# UNIQUE: "home.dest","boat"
# print(df2.describe())
# print(df2.info())
df2.dropna(inplace=True, subset=['survived','pclass','sex','age'])
df2['sanitized'] = df2['name'].replace(r'[' + string.punctuation + r']', '', regex=True)
df2 = df2.sort_values(by='sanitized')
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df2['sanitized'])
# print(df2)
# print(len(df2))

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

# merged = pd.merge(df1, df2, on=['sanitized'], how='inner')
merged = []
matched = 0
unmatched = 0
for row1 in df1.itertuples():
    # print(row1.Index, row1.sanitized)
    match1 = re.search(r'^(\w+ \w+ \w+)', row1.sanitized)
    found = 0
    for row2 in df2.itertuples():
        match2 = re.search(r'^(\w+ \w+ \w+)', row2.sanitized)
        if match1.group(1) == match2.group(1):
            matched += 1
            # print(matched, row2.sanitized)
            found = 1
            # print(row2)
            row = pd.DataFrame([row2])
            # print(row.info())
            row['SibSp'] = None
            row['Parch'] = None
            row['Fare'] = None
            # print(len(row))
            d = row.to_dict('records')
            merged.append(d[0])
            break
    if not found:
        unmatched += 1
        # print('*', unmatched, row1.sanitized)
        found = 0
        # print(row1)
        row = pd.DataFrame([row1])
        # print(row)
        row['Boat'] = None
        row['Home'] = None
        # print(row)
        # print(row.info())
        # print(len(row))
        d = row.to_dict('records')
        # print(d)
        merged.append(d[0])
        # print(merged[row1.sanitized])
# print(merged)
df3 = pd.DataFrame(merged)
df3.drop(columns=['sanitized'], inplace=True)
print(df3)

# df = pd.get_dummies(df, columns=['Pclass', 'Sex', 'Embarked', 'home.dest', 'boat'])

# # Split the data into training and test sets
# X = df.drop(['survived', 'name', 'ticket', 'room'], axis=1)
# y = df['survived']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Build the model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)

# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)

# Pandas(Index=153, PassengerId=154, Survived=0, Pclass=3, Name='van Billiard, Mr. Austin Blyler', Sex='male', Age=40.5, Embarked='S', Cabin=nan, Ticket='A/5. 851', sanitized='van Billiard Mr Austin Blyler', SibSp=0, Parch=2, Fare=14.5)
# Pandas(Index=388, _1=389, survived=0, pclass='2nd', name='del Carlo, Mr Sebastiano', sex='male', age=28.0, embarked='Cherbourg', room=nan, ticket=nan, sanitized='del Carlo Mr Sebastiano', _7='Lucca, Italy / California', boat='(295)')
