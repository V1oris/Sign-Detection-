import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

train = pd.read_csv('train.csv')

X = train.drop('label', axis=1)
y = train['label']

model = RandomForestClassifier(random_state=0)
model.fit(X,y)

print(train)
f = open('model.p', 'wb')
pickle.dump({'model' : model}, f)
f.close()