import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, max_error, \
    mean_squared_log_error, median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

'''
1. 读取数据
2. 建立模型
3. 模型的训练
4. 模型的预测
'''

"""
2.1 经过查询，无序多分类无法直接引入，必须“哑元”化变量
2.2 等级变量（有序多分类）可以直接引入模型，也就是我们需要将变量进行有序数字化
2.3 数字化的结果才放到结果里面进行预测
"""
df = pd.read_csv('../二手房数据/data_all.csv', encoding='utf-8')

print(df)

# 查看数据类型
# print(df.dtypes)


# 查看唯一标签值
# print(df['房屋所属市辖区'].unique())
# print(df['房屋地址（街道）'].unique())
# print(df['所在楼层'].unique())
# print(df['建筑面积（平方米）'].unique())
# print(df['户型结构'].unique())
# print(df['建筑类型'].unique())
# print(df['房屋朝向'].unique())
# print(df['建筑结构'].unique())
# print(df['装修情况'].unique())
# print(df['配备电梯'].unique())
# print(df['挂牌时间'].unique())
# print(df['装修情况'].unique())
# print(df['交易权属'].unique())
# print(df['房屋用途'].unique())

# 判断朝向，用空格分隔取第一个
df['房屋朝向'] = df.房屋朝向.apply(lambda x: x.split(' ')[0])

# 判断楼层，只要前面三个字符
df['所在楼层'] = df.所在楼层.apply(lambda x: x.split('(')[0])

print(df['房屋所属市辖区'].unique())
print(df['房屋地址（街道）'].unique())
print(df['所在楼层'].unique())
print(df['房屋户型'].unique())
print(df['建筑面积（平方米）'].unique())
print(df['户型结构'].unique())
print(df['建筑类型'].unique())
print(df['房屋朝向'].unique())
print(df['建筑结构'].unique())
print(df['装修情况'].unique())
print(df['配备电梯'].unique())
print(df['挂牌时间'].unique())
print(df['装修情况'].unique())
print(df['交易权属'].unique())
print(df['房屋用途'].unique())


# 拆解户型， 把对应的数字分出来，新增对应的列进去

def get_shi(x):
    return int(x.split('室')[0])


def get_ding(x):
    return int(x.split('厅')[0].split('室')[1])


def get_chu(x):
    return int(x.split('厨')[0].split('厅')[1])


def get_wei(x):
    return int(x.split('卫')[0].split('厨')[1])


df['室'] = df['房屋户型'].map(get_shi)
df['厅'] = df['房屋户型'].map(get_ding)
df['厨'] = df['房屋户型'].map(get_chu)
df['卫'] = df['房屋户型'].map(get_wei)

df['总价'] = df['单价（元/平方米）'] * df['建筑面积（平方米）']

# print(df)

# 单独保存原数据，删除所有的空值
d1 = df.dropna().reset_index(drop=True)
# 删掉楼层、单价、户型、房屋地址（街道）

d1.drop(columns=['房屋户型', '所在楼层', '单价（元/平方米）', '房屋地址（街道）', '挂牌时间'], inplace=True)

# print(d1)

# 有序化所有的多分类标签

print(d1.dtypes)
'''
1. 朝向
'''
map1 = {'南': 5, '南北': 6, '北': 1, '西南': 10, '东西': 4, '东': 2, '东北': 8, '东南': 9, '西': 3, '西北': 7}
d1['房屋朝向'] = d1['房屋朝向'].map(map1)

'''
2. 装修情况
'''

map2 = {'毛坯': 1, '精装': 2, '简装': 3, '其他': 4}
d1['装修情况'] = d1['装修情况'].map(map2)
'''
3. 配备电梯 
'''
map3 = {'有': 1, '无': 0}
d1['配备电梯'] = d1['配备电梯'].map(map3)
'''
4. 交易权属
'''
map4 = {'拆迁安置房': 6, '商品房': 5, '经济适用房': 7, '已购公房': 4, '限价商品房': 3, '集资房': 8, }
d1['交易权属'] = d1['交易权属'].map(map4)
'''
5. 房屋用途
'''
map5 = {'普通住宅': 4, '商业办公类': 3, '公寓': 1, '商住楼': 2, '酒店式公寓': 5}
d1['房屋用途'] = d1['房屋用途'].map(map5)

'''
6. 户型结构
'''
map6 = {'平层': 4, '开间': 2, '跃层': 5, '错层': 1, '复式': 3}
d1['户型结构'] = d1['户型结构'].map(map6)

'''
7. 建筑类型
'''
map7 = {'板楼': 4, '板塔结合': 3, '塔楼': 2, '平房': 1, }
d1['建筑类型'] = d1['建筑类型'].map(map7)

'''
8. 建筑结构
'''
map8 = {'砖混结构': 6, '框架结构': 5, '钢混结构': 7, '混合结构': 4, '未知结构': 2, '砖木结构': 3, '钢结构': 8}
d1['建筑结构'] = d1['建筑结构'].map(map8)
'''
9. 房屋所属市辖区
'''
map9 = {'成华': 1, '崇州': 2, '都江堰': 3, '高新': 4, '高新西': 5, '简阳': 6, '锦江': 7, '金牛': 8, '龙泉驿': 9, '彭州': 10, '郫都': 11,
        '青白江': 12,
        '青羊': 13, '双流': 14, '天府新区': 15, '温江': 16, '武侯': 17, '新都': 18, '新津': 19, '金堂': 20}
d1['房屋所属市辖区'] = d1['房屋所属市辖区'].map(map9)

print(d1)

# 保存表量化的数据

d1.to_csv('../二手房数据/house.csv', encoding='utf-8', index=False)

'''

'''

X = d1.drop(columns=['总价'])
y = d1['总价']
# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
# 而预测值为高次关系，我们可以使用PolynomialFeatures生成高次的特征去更好拟合预测值
poly = PolynomialFeatures(degree=2)
# 标准化
x_train = poly.fit_transform(X_train.values)
x_test = poly.fit_transform(X_test)
print(X, y)

"""
3. 模型的训练

3.1 线性回归
3.2 随机森林
3.3 决策树
3.4 k近邻
"""

'''
3.1 线性回归
'''
# 线性回归
# Lasso回归有时也叫做线性回归的L1正则化，和Ridge回归的主要区别就是在正则化项，Ridge回归用的是L2正则化，而Lasso回归用的是L1正则化。
# alpha :正则项系数，初始值为1，数值越大，则对复杂模型的惩罚力度越大。

print('==============Lasso线性回归======================================================================')
print("正则项系数:0.1")
la = Lasso(alpha=0.1, fit_intercept=True, normalize=True,
           precompute=False, copy_X=True, max_iter=1000, tol=1e-4,
           warm_start=False, positive=False, random_state=None,
           selection='cyclic')
la.fit(x_train, y_train)
la_y_predict = la.predict(x_test)
#
print(f'训练集得分：{round(la.score(x_train, y_train), 2)}')
print(f'测试集得分：{round(la.score(x_test, y_test), 2)}')
print("使用的特性数量:{}".format(np.sum(la.coef_ != 0)))
print("正则项系数:0.01")
la1 = Lasso(alpha=50, fit_intercept=True, normalize=True,
            precompute=False, copy_X=True, max_iter=1000, tol=1e-4,
            warm_start=False, positive=False, random_state=None,
            selection='cyclic')
la1.fit(x_train, y_train)
print(f'训练集得分：{round(la1.score(x_train, y_train), 2)}')
print(f'测试集得分：{round(la1.score(x_test, y_test), 2)}')
print("使用的特性数量:{}".format(np.sum(la1.coef_ != 0)))

# 结果评估
# print('Lasso线性回归的均方误差为 :', np.sqrt(mean_squared_error(y_test, la_y_predict)))  # 计算均方差根判断效果
print('Lasso线性回归的平均绝对误差为:', r2_score(y_test, la_y_predict))  # 计算均方误差回归损失，越接近于1拟合效果越好
#
#
#
# # 绘制训练集和测试集相似度曲线
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 绘图展示预测效果
# la_y_predict.tolist().sort()
# y_test.tolist().sort()
# x = np.arange(1, 3711)
# Pplot = plt.scatter(x, la_y_predict)
# Tplot = plt.scatter(x, y_test)
# plt.legend(handles=[Pplot, Tplot], labels=['y_pred', 'y_test'])
# plt.show()

'''
3.2 随机森林
'''
print('==============随机森林======================================================================')
from sklearn.metrics import classification_report
rf = RandomForestRegressor(n_estimators=20, max_features=0.4, max_depth=15)
# 训练
rf.fit(x_train, y_train)
dtr_y_predict = rf.predict(x_test)
# 模型的评估
print(f'训练集得分：{round(rf.score(x_train, y_train), 2)}')
print(f'测试集得分：{round(rf.score(x_test, y_test), 2)}')
# print('随机森林的均方误差为 :', np.sqrt(mean_squared_error(y_test, dtr_y_predict)))  # 计算均方差根判断效果
print('随机森林的平均绝对误差为:', r2_score(y_test, dtr_y_predict))  # 计算均方误差回归损失，越接近于1拟合效果越好
print('解释方差回归得分：\n',explained_variance_score(y_test, dtr_y_predict)) #解释方差回归得分
print('最大剩余误差：\n',max_error(y_test, dtr_y_predict))  # 最大剩余误差
print('平均绝对误差回归损失：\n',mean_absolute_error(y_test, dtr_y_predict)) #平均绝对误差回归损失
print('均方误差回归损失：\n',mean_squared_error(y_test, dtr_y_predict)) #均方误差回归损失
print('均方对数误差回归损失：\n',mean_squared_log_error(y_test, dtr_y_predict))#均方对数误差回归损失
print('中位绝对误差回归损失：\n',median_absolute_error(y_test, dtr_y_predict) ) #中位绝对误差回归损失
#
# 绘制训练集和测试集相似度曲线
import numpy as np
import matplotlib.pyplot as plt

# 绘图展示预测效果
dtr_y_predict.tolist().sort()
y_test.tolist().sort()
print(len(dtr_y_predict.tolist()))
print(len(y_test.tolist()))
x = np.arange(1, 3711)
Pplot = plt.scatter(x, dtr_y_predict)
Tplot = plt.scatter(x, y_test)
plt.legend(handles=[Pplot, Tplot], labels=['y_pred', 'y_test'])
plt.show()



# print('--------------测试 RandomForestRegressor 的预测性能随 n_estimators 参数的影响--------------')
# def test_RandomForestRegressor_num(*data):
#     '''
#     测试 RandomForestRegressor 的预测性能随  n_estimators 参数的影响
#     '''
#     X_train, X_test, y_train, y_test = data
#     nums = np.arange(1, 100, step=2)
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     testing_scores = []
#     training_scores = []
#     for num in nums:
#         regr = RandomForestRegressor(n_estimators=num)
#         regr.fit(X_train, y_train)
#         training_scores.append(regr.score(X_train, y_train))
#         testing_scores.append(regr.score(X_test, y_test))
#     ax.plot(nums, training_scores, label="Training Score")
#     ax.plot(nums, testing_scores, label="Testing Score")
#     ax.set_xlabel("estimator num")
#     ax.set_ylabel("score")
#     ax.legend(loc="lower right")
#     ax.set_ylim(-1, 1)
#     plt.suptitle("RandomForestRegressor")
#     plt.show()
#
#
# # 调用 test_RandomForestRegressor_num
# test_RandomForestRegressor_num(x_train, x_test, y_train, y_test)
#
# print('--------------测试 RandomForestRegressor 的预测性能随 max_depth 参数的影响--------------')
# def test_RandomForestRegressor_max_depth(*data):
#     '''
#     测试 RandomForestRegressor 的预测性能随  max_depth 参数的影响
#     '''
#     X_train, X_test, y_train, y_test = data
#     maxdepths = range(1, 20)
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     testing_scores = []
#     training_scores = []
#     for max_depth in maxdepths:
#         regr = RandomForestRegressor(max_depth=max_depth)
#         regr.fit(X_train, y_train)
#         training_scores.append(regr.score(X_train, y_train))
#         testing_scores.append(regr.score(X_test, y_test))
#     ax.plot(maxdepths, training_scores, label="Training Score")
#     ax.plot(maxdepths, testing_scores, label="Testing Score")
#     ax.set_xlabel("max_depth")
#     ax.set_ylabel("score")
#     ax.legend(loc="lower right")
#     ax.set_ylim(0, 1.05)
#     plt.suptitle("RandomForestRegressor")
#     plt.show()
#
#
# # 调用 test_RandomForestRegressor_max_depth
# test_RandomForestRegressor_max_depth(x_train, x_test, y_train, y_test)
#
# print('--------------测试 RandomForestRegressor 的预测性能随 max_features 参数的影响--------------')
# def test_RandomForestRegressor_max_features(*data):
#     '''
#    测试 RandomForestRegressor 的预测性能随  max_features 参数的影响
#     '''
#     X_train, X_test, y_train, y_test = data
#     max_features = np.linspace(0.01, 1.0)
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     testing_scores = []
#     training_scores = []
#     for max_feature in max_features:
#         regr = RandomForestRegressor(max_features=max_feature)
#         regr.fit(X_train, y_train)
#         training_scores.append(regr.score(X_train, y_train))
#         testing_scores.append(regr.score(X_test, y_test))
#     ax.plot(max_features, training_scores, label="Training Score")
#     ax.plot(max_features, testing_scores, label="Testing Score")
#     ax.set_xlabel("max_feature")
#     ax.set_ylabel("score")
#     ax.legend(loc="lower right")
#     ax.set_ylim(0, 1.05)
#     plt.suptitle("RandomForestRegressor")
#     plt.show()
#
#
# # 调用 test_RandomForestRegressor_max_features
# test_RandomForestRegressor_max_features(x_train, x_test, y_train, y_test)
#


'''
3.3 决策树
'''
# print('==============决策树======================================================================')
dt = DecisionTreeRegressor(criterion='mse', max_depth=9, max_features='sqrt', min_samples_split=2,
                           min_samples_leaf=1, random_state=0)
y_pred_dt = dt.fit(x_train, y_train)
dt_y_predict = dt.predict(x_test)
# print(f'训练集得分：{round(dt.score(x_train, y_train), 2)}')
# print(f'测试集得分：{round(dt.score(x_test, y_test), 2)}')

# criterion='mse' ,max_depth=None,max_features='sqrt',min_samples_split=2,min_samples_leaf=1,random_state=0


# print('Lasso线性回归的均方误差为 :', np.sqrt(mean_squared_error(y_test, dt_y_predict)))  # 计算均方差根判断效果
# print('决策树回归的平均绝对误差为:', r2_score(y_test, dt_y_predict))  # 计算均方误差回归损失，越接近于1拟合效果越好
#
# # 绘制训练集和测试集相似度曲线
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 绘图展示预测效果
# dt_y_predict.tolist().sort()
# y_test.tolist().sort()
# x = np.arange(1, 3711)
# Pplot = plt.scatter(x, dt_y_predict)
# Tplot = plt.scatter(x, y_test)
# plt.legend(handles=[Pplot, Tplot], labels=['y_pred', 'y_test'])
# plt.show()

'''
3.4 k近邻
'''
print('==============k近邻======================================================================')

kn = KNeighborsRegressor(n_neighbors=20)
kn.fit(x_train, y_train)
y_pred = kn.predict(x_test)
print(f'训练集得分：{round(kn.score(x_train, y_train), 2)}')
print(f'测试集得分：{round(kn.score(x_test, y_test), 2)}')
# # print('k近邻的均方误差为 :', np.sqrt(mean_squared_error(y_test, y_pred)))  # 计算均方差根判断效果
# print('k近邻的平均绝对误差为:', r2_score(y_test, y_pred))  # 计算均方误差回归损失，越接近于1拟合效果越好
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 绘图展示预测效果
# y_pred.tolist().sort()
# y_test.tolist().sort()
# print(len(y_pred.tolist()))
# print(len(y_test.tolist()))
# x = np.arange(1, 3711)
# Pplot = plt.scatter(x, y_pred)
# Tplot = plt.scatter(x, y_test)
# plt.legend(handles=[Pplot, Tplot], labels=['y_pred', 'y_test'])
# plt.show()

"""
4. 模型的预测
"""

'''
4.1 现在买房要求为 :

1. 在天府新区购买(15)
2. 面积大概再105㎡左右（105）
3. 户型结构：平层（4）
4. 建筑类型：平房（1）
5. 房屋朝向：南（5）
6. 建筑结构：钢混结构（7）
7. 装修情况：精装（2）
8. 配备电梯：有（1）
9. 交易权属：商品房（5）
10.房屋用途：普通住宅（4）
11. 3室1厅1厨1卫（3、1、1、1）
'''

apply = np.array([15, 105, 4, 1, 5, 7, 2, 1, 5, 4, 3, 1, 1, 1]).reshape(1, -1)
poly_apply = poly.fit_transform(apply)
print('------------总价预测结果-------------')
print(f'线性回归：{round(la.predict(poly_apply)[0], 2)}元')
print(f'随机森林：{round(rf.predict(poly_apply)[0], 2)}元')
print(f'决策树：{round(dt.predict(poly_apply)[0], 2)}元')
print(f'K近邻：{round(kn.predict(poly_apply)[0], 2)}元')
print('------------综合预测结果-------------')
print(round(((la.predict(poly_apply) + rf.predict(poly_apply) + dt.predict(poly_apply) + kn.predict(
    poly_apply) ) / 4.0)[0], 2), '元')
