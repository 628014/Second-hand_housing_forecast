import csv
from collections import Counter
from pyecharts.charts import Funnel, Bar
import pandas as pd
from pyecharts.globals import ThemeType
from pyecharts.charts import Map
import os

from pyecharts.render import snapshot, make_snapshot

df = pd.read_csv('../二手房数据/data_all.csv', encoding='utf-8')
# 装修情况 单价（元/平方米）
from pyecharts import options as opts
from pyecharts.charts import Pie

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
theme = ThemeType.ESSOS
"""
探究装修情况和单价的关系 ：南丁格尔
"""
decorating_class = []
decoration = df.groupby(by=["装修情况"])["单价（元/平方米）"].median().sort_values(ascending=False).to_dict()
decorating_class = [[k, v] for k, v in decoration.items()]
pie1 = Pie({"theme": theme}).add("装修情况和总价的关系",  # 添加提示框标签
                                 decorating_class,  # 输入数据
                                 radius=["20%", "70%"],  # 设置内半径和外半径
                                 center=["50%", "50%"],  # 设置圆心位置
                                 rosetype="radius")  # 玫瑰图模式，通过半径区分数值大小，角度大小表示占比
pie1.set_global_opts(title_opts=opts.TitleOpts(title="装修和单价的关系",  # 设置图标题
                                               pos_right='50%'),  # 图标题的位置
                     legend_opts=opts.LegendOpts(  # 设置图例|
                         orient='vertical',  # 垂直放置图例
                         pos_right="90%",  # 设置图例位置
                         pos_top="10%"))

pie1.set_series_opts(label_opts=opts.LabelOpts(formatter="{b} : {d}%"))  # 设置标签文字形式为（国家：占比（%））
pie1.render("装修&单价.html")

"""
探究所在楼层和单价的关系 ： 饼图
"""
df["所在楼层"] = df["所在楼层"].map(lambda x: x[0:3])
decorating_class = []
decoration = df.groupby(by=["所在楼层"])["单价（元/平方米）"].median().sort_values(ascending=False).to_dict()
decorating_class = [[k, v] for k, v in decoration.items()]
pie1 = Pie({"theme": theme}).add("所在楼层和单价的关系",  # 添加提示框标签
                                 decorating_class,  # 输入数据
                                 radius=["0%", "70%"],  # 设置内半径和外半径
                                 center=["50%", "50%"],  # 设置圆心位置
                                 rosetype="radius")  # 玫瑰图模式，通过半径区分数值大小，角度大小表示占比
pie1.set_global_opts(title_opts=opts.TitleOpts(title="装修和单价的关系",  # 设置图标题
                                               pos_right='50%'),  # 图标题的位置
                     legend_opts=opts.LegendOpts(  # 设置图例|
                         orient='vertical',  # 垂直放置图例
                         pos_right="90%",  # 设置图例位置
                         pos_top="10%"))

pie1.set_series_opts(label_opts=opts.LabelOpts(formatter="{b} : {d}%"))  # 设置标签文字形式为（国家：占比（%））
pie1.render("楼层&单价.html")

"""
探究电梯和总价的关系
"""
elevator_count = df.groupby(by=["配备电梯"])["单价（元/平方米）"].median().sort_values(ascending=False)
count = Counter(df["配备电梯"])
pie_x = ["有", "无"]
pie_y = elevator_count.tolist()
pie = (
    Pie({"theme": theme})
        .add("", [list(z) for z in zip(pie_x, pie_y)])
        .set_global_opts(title_opts=opts.TitleOpts(title="是否配备电梯和单价的关系"))
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c} ({d}%)"))
)
pie.render("电梯&单价.html")

"""
房屋用途 & 单价
"""
home_category = df.groupby(by=["房屋用途"])["单价（元/平方米）"].median().sort_values(ascending=False)
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
c_d1.render("住宅类别&单价.html")
"""
建筑结构 & 单价
"""
structure = df.groupby(by=["建筑结构"])["单价（元/平方米）"].median().sort_values(ascending=False)
structureX = structure.index.to_list()
structureY = structure.to_list()
data = [[structureX[i], structureY[i]] for i in range(len(structureX))]

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
funnel.render("建筑结构&单价.html")
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
c_d1.render("建筑类别&单价.html")


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
c = (
    Map({"theme": theme})
        .add("成都", [list(z) for z in zip(area_dict.keys(), area_dict.values())], "成都")
        .set_global_opts(
        title_opts=opts.TitleOpts(title="成都地图"),
        visualmap_opts=opts.VisualMapOpts(
            is_show=True,  # 视觉映射配置
            max_=max(area_dict.values()),
            is_calculable=True,
        )
    ).render("成都地图.html")
)