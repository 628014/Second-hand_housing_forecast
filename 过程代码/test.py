import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#加载数据集
boston = load_boston()
print(boston.data.shape)#boston是sklearn.utils.bunch类型，里面有data506行13列
X = boston.data
y = boston.target
#筛选和标签最相关的k=5个特征
selector = SelectKBest(f_regression,k=4)
X_new = selector.fit_transform(X,y)
print(X_new.shape)
print(selector.get_support(indices=True).tolist())#查看最相关的是那几列
#划分数据集
X_train,X_test,y_train,y_test = train_test_split(X_new,y,test_size=0.3,random_state=666)
#print(X_train.shape,y_train.shape)
#均值方差归一化
standardscaler = StandardScaler()
standardscaler.fit(X_train)
X_train_std = standardscaler.transform(X_train)
X_test_std = standardscaler.transform(X_test)
#训练
kNN_reg = KNeighborsRegressor()
kNN_reg.fit(X_train_std,y_train)
#预测
y_pred = kNN_reg.predict(X_test_std)
print(np.sqrt(mean_squared_error(y_test, y_pred)))#计算均方差根判断效果
print(r2_score(y_test,y_pred))#计算均方误差回归损失，越接近于1拟合效果越好

#绘图展示预测效果
print(type(y_pred))
y_pred.sort()
y_test.sort()
print(len(y_pred.tolist()))
print(len(y_test.tolist()))
x = np.arange(1,153)
Pplot = plt.scatter(x,y_pred)
Tplot = plt.scatter(x,y_test)
plt.legend(handles=[Pplot,Tplot],labels=['y_pred','y_test'])
plt.show()