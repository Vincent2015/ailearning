import pandas as pd
import pickle
df_ads = pd.read_csv("易速鲜花微信软文.csv")

df_ads.dropna(inplace=True)
df_ads['转发数'].fillna(df_ads['转发数'].mean(),inplace=True)

#X = df_ads.drop(['浏览量','热度指数','文章评级' ],axis=1)
X=df_ads.iloc[:,0:2]
print(X)
y = df_ads.浏览量

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state =0)

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression(fit_intercept = True)

linear_model.fit(X_train,y_train)
y_pred = linear_model.predict(X_test)

print('当前模型的4个特征的权重分别是: ', linear_model.coef_)
print('当前模型的截距（偏置）是: ', linear_model.intercept_)
print("模型评分",linear_model.score(X_test,y_test))

pickle.dump(linear_model, open('model.pkl','wb')) #序列化模型（就是存盘）
linear_model = pickle.load(open('model.pkl','rb')) #反序列化模型（就是再导入模型）