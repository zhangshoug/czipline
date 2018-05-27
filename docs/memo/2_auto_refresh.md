
`cswd`从网易、巨潮等公共网站提取A股数据，包括：
+ 股票日线
+ 指数日线
+ 行业
+ 概念
+ 地域
+ 财务报告及财务指标
+ 融资融券
+ 国库券资金成本

+ `websource`模块主要负责提取网络数据，分网站划分。如`websource.wy`从网易采集数据。模块下函数以`fetch_**`命名，代表提取某类数据。

+ `sqldata`负责存储和更新数据。
    + 每个数据表如为空，则初始化该表
    + 如非空，则逐个代码找出其在该表的最后一日，以次日开始刷新

+ `dataproxy`处理缓存。与`functools`等缓存机制不同的是，缓存可以存活至指定时点，尽管会话此时已经结束。
    + 默认失效时点为当天18点，也可根据实际情形予以设定。此外，如果缓存时长超过24小时，数据失效。
    + 类`read()`方法首先查看本地文件最新修改时间，如果失效，则重新从网络采集。
    + 使用时请注意采集函数默认值`None`的处理，建议使用关键字指定函数参数，避免出错。

借助`windows`任务计划程序，可以定期自动处理数据更新。`task_schedules`给出了参考模板，可以根据自己需要更改使用。
+ 在设定后台自动处理前，最好先运行该脚本，观察运行结果是否符合预期
+ 每日刷新数据用时约为30分钟
+ 设定任务计划参考图

+ 是否需要查看运行结果
![如果希望后台运行，选择`不管用户是否登陆都要运行`](https://github.com/liudengfeng/zipline/blob/master/docs/memo/images/task_1.PNG)

+ 设定运行计划
![设定运行计划](https://github.com/liudengfeng/zipline/blob/master/docs/memo/images/task_2.PNG)
+ 设定操作
    + `添加参数`为要运行的脚本文件路径，请加双引号
    + `起始于`为要运行的环境，请指定为`zipline`所安装的环境目录。一般为`C:\Users\<用户名>\Anaconda3\envs\<环境名>`
![设定操作](https://github.com/liudengfeng/zipline/blob/master/docs/memo/images/task_3.PNG)

+ 类似`ingest`处理脚本[请参考](https://github.com/liudengfeng/zipline/blob/master/task_schedules/prepare.py)
