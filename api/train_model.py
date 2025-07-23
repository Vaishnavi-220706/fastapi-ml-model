import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
data={
    'math':[78,65,78,45,30,60,80,40,20,50],
    'science':[20,68,45,82,63,81,20,40,60,90],
    'english':[30,20,45,65,98,42,30,30,40,60],
    'result':['fail','pass','fail','fail','fail','pass','fail','fail','fail','pass']
}
df=pd.DataFrame(data)
df['result']=df['result'].map({'pass':1,'fail':0})
#train the model
x=df[['math','science','english']]
y=df['result']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
#call the model
model=LogisticRegression()
model.fit(x_train,y_train)
joblib.dump(model,'model.pkl')