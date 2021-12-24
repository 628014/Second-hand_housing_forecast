from collections import Counter

import pandas as pd
import seaborn as sns
from pyecharts.charts import Funnel, Bar, HeatMap
from pyecharts.charts import Map
from pyecharts.globals import ThemeType

df = pd.read_csv('../二手房数据/data_all.csv', encoding='utf-8')
# 装修情况 单价（元/平方米）
from pyecharts import options as opts
from pyecharts.charts import Pie

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
theme = ThemeType.ESSOS
"""
探究装修情况数目和房屋：南丁格尔
"""
decorating_class = []
decoration = df.groupby(by=["装修情况"])["单价（元/平方米）"].sum().sort_values(ascending=False).to_dict()
decorating_class = [[k, v] for k, v in decoration.items()]
pie1 = Pie({"theme": theme}).add("装修情况平均值",  # 添加提示框标签
                                 decorating_class,  # 输入数据
                                 radius=["20%", "70%"],  # 设置内半径和外半径
                                 center=["50%", "50%"],  # 设置圆心位置
                                 rosetype="radius")  # 玫瑰图模式，通过半径区分数值大小，角度大小表示占比
pie1.set_global_opts(title_opts=opts.TitleOpts(title="装修情况和单价的关系",  # 设置图标题
                                               pos_right='50%'),  # 图标题的位置
                     legend_opts=opts.LegendOpts(  # 设置图例|
                         orient='vertical',  # 垂直放置图例
                         pos_right="90%",  # 设置图例位置
                         pos_top="10%"))

pie1.set_series_opts(label_opts=opts.LabelOpts(formatter="{b} : {d}"))  # 设置标签文字形式为（国家：占比（%））
pie1.render("./hjy_pic/装修类别数目和房屋：南丁格尔.html")

decorationMed = df.groupby(by=["装修情况"])["单价（元/平方米）"].median().sort_values(ascending=False)
decoration_bar = (
    Bar({"theme": theme})
        .add_xaxis(decorationMed.index.to_list())
        .add_yaxis("", decorationMed.to_list())
        .set_global_opts(
        title_opts={"text": "装修情况与单价的关系", "subtext": ""}
    )
)
decoration_bar.render("./hjy_pic/装修情况与单价：柱状图.html")
"""
探究所在楼层和单价的关系 ： 饼图
"""
df["所在楼层"] = df["所在楼层"].map(lambda x: x[0:3])
decorating_class = []
decoration = df.groupby(by=["所在楼层"])["单价（元/平方米）"].sum().sort_values(ascending=False).to_dict()
decorating_class = [[k, v] for k, v in decoration.items()]
pie1 = Pie({"theme": theme}).add("所在楼层和数量的关系",  # 添加提示框标签
                                 decorating_class,  # 输入数据
                                 radius=["0%", "70%"],  # 设置内半径和外半径
                                 center=["50%", "50%"],  # 设置圆心位置
                                 rosetype="radius")  # 玫瑰图模式，通过半径区分数值大小，角度大小表示占比
pie1.set_global_opts(title_opts=opts.TitleOpts(title="所在楼层和数量的关系",  # 设置图标题
                                               pos_right='50%'),  # 图标题的位置
                     legend_opts=opts.LegendOpts(  # 设置图例|
                         orient='vertical',  # 垂直放置图例
                         pos_right="90%",  # 设置图例位置
                         pos_top="10%"))

pie1.set_series_opts(label_opts=opts.LabelOpts(formatter="{b} : {d}%"))  # 设置标签文字形式为（国家：占比（%））
pie1.render("./hjy_pic/所在楼层&数量.html")

floor = df.groupby(by=["所在楼层"])["单价（元/平方米）"].mean().sort_values(ascending=False)
pie1 = (
    Bar({"theme": theme})
        .add_xaxis(floor.index.to_list())
        .add_yaxis("", floor.to_list())
        .set_global_opts(
        title_opts={"text": "所在楼层与单价的关系", "subtext": ""}
    ))
pie1.set_global_opts(
    title_opts=opts.TitleOpts(title="所在楼层和单价的关系",  # 设置图标题
                              pos_right='50%'),  # 图标题的位置
    legend_opts=opts.LegendOpts(  # 设置图例|
        orient='vertical',  # 垂直放置图例
        pos_right="90%",  # 设置图例位置
        pos_top="10%"))

pie1.set_series_opts(label_opts=opts.LabelOpts(formatter="{b} : {c}"))  # 设置标签文字形式为（国家：占比（%））
pie1.render("./hjy_pic/所在楼层&单价：柱状图.html")

"""
探究电梯和总价的关系
"""


# elevator_equip = df.groupby(by=["配备电梯"])


def floorChange(x):
    if x == "高楼层":
        return "高楼层"
    else:
        return "中低楼层"


df["楼层"] = df["所在楼层"].map(floorChange)
elevator_equip = df.groupby(by=["楼层"])["配备电梯"].sum()
highDic = Counter(elevator_equip["中低楼层"])
lowDic = Counter(elevator_equip["高楼层"])
x_axis = ["中低楼层", "高楼层"]
y_axis = ["无", "有"]
data = [[0, 0, lowDic["无"]], [0, 1, lowDic["有"]], [1, 0, highDic["无"]], [1, 1, highDic["有"]]]
print(data)
heatmap = (
    HeatMap()
        .add_xaxis(x_axis)
        .add_yaxis("有无电梯", y_axis, data)
        .set_global_opts(
        title_opts=opts.TitleOpts(title="不同楼层与电梯关系热力图"),
        visualmap_opts=opts.VisualMapOpts(max_=highDic["有"], min_=lowDic["无"])
    )
)
heatmap.render("./hjy_pic/电梯&单价.html")

"""
房屋用途 & 单价
"""


def use(x):
    if x == "普通住":
        return "普通住宅"
    else:
        return x


df["房屋用途"] = df["房屋用途"].map(use)
home_category = df.groupby(by=["房屋用途"])["单价（元/平方米）"].mean()
home_categoryX = home_category.index.to_list()
home_categoryY = home_category.to_list()

c_d1 = (
    Bar({"theme": theme})
        .add_xaxis(home_categoryX)
        .add_yaxis("", home_categoryY)
        .set_global_opts(
        title_opts={"text": "房屋用途与单价的关系", "subtext": ""}
    )
)
c_d1.render("./hjy_pic/房屋用途&单价.html")
"""
建筑结构 & 单价
"""
structure = df.groupby(by=["建筑结构"])["单价（元/平方米）"].mean().sort_values(ascending=False)
structureX = structure.index.to_list()
structureY = structure.to_list()
data = [[structureX[i], structureY[i]] for i in range(len(structureX))]
print(structure)
funnel = (
    Funnel({"theme": theme}).add(
        series_name="",
        data_pair=data,
        gap=2,
        tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{a} <br/>{b} : {c}"),
        label_opts=opts.LabelOpts(is_show=True, position="inside"),
        itemstyle_opts=opts.ItemStyleOpts(border_color="#fff", border_width=1),
    ).set_global_opts(title_opts=opts.TitleOpts(title="建筑结构-单价", subtitle=""))
)
funnel.render("./hjy_pic/建筑结构&单价.html")
"""
建筑类型 & 单价
"""
building_type = df.groupby(by=["建筑类型"])["单价（元/平方米）"].median().sort_values(ascending=False)
building_typeX = building_type.index.to_list()
building_typeY = building_type.to_list()
c_d1 = (
    Bar({"theme": theme})
        .add_xaxis(building_typeX)
        .add_yaxis("", building_typeY)
        .set_global_opts(
        title_opts={"text": "建筑类别与单价的关系", "subtext": ""}
    )
)
c_d1.render("./hjy_pic/建筑类别&单价.html")


# 基础数据
def change(x):
    if x == "高新西":
        x = "高新"
    if x == "天府新区":
        return x
    if x == "金堂" or x == "大邑" or x == "蒲江" or x == "新津":
        x += "县"
    elif x == "简阳" or x == "都江堰" or x == "彭州" or x == "崇州" or x == "邛崃":
        x += "市"
    else:
        x += "区"

    return x


"""
地图
"""
df["房屋所属市辖区"] = df["房屋所属市辖区"].map(change)
area_dict = Counter(df["房屋所属市辖区"])
print(area_dict)
c = (
    Map({"theme": theme}).add("成都", [list(z) for z in zip(area_dict.keys(), area_dict.values())], "成都")
        .set_global_opts(
        title_opts=opts.TitleOpts(title="成都地图"),
        visualmap_opts=opts.VisualMapOpts(
            is_show=True,  # 视觉映射配置
            max_=max(area_dict.values()),
            is_calculable=True,
        )
    ).render("./hjy_pic/成都地图.html")
)
