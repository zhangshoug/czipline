
# 前置条件
+ 新建`zipline`环境（名称随意，以下均假定在此环境下安装）
+ 安装`cswd`数据包
+ 确认Visual Studio Installer选项
	+ ![C编译器](https://github.com/liudengfeng/zipline/blob/master/docs/memo/images/installer_1.PNG)
	+ ![本机开发工具](https://github.com/liudengfeng/zipline/blob/master/docs/memo/images/installer_2.PNG)

# 依赖包

## `pip`安装
+ `pip安装requirements.txt`所列包
+ 如有安装失败，请下载对应whl包
    + [下载whl网址](https://www.lfd.uci.edu/~gohlke/pythonlibs)
    + ![参考whl清单](https://github.com/liudengfeng/zipline/blob/master/docs/memo/images/whl_packages.PNG)

# 安装`zipline`
+ `clone`项目到本地
+ 使用模式
    + 进入`zipline`环境
    + 进入`setup.py`所在的目录
    + `python setup.py install`
+ 开发模式
    + 进入`zipline`环境
    + 进入`setup.py`所在的目录
    + `python setup.py build_ext --inplace`
    + `python setup.py develop`
+ 重新安装本地`odo`和`blaze`
	+ 移除原有`odo`及`blaze`
	+ 安装改版`odo`和`blaze`


+ **注意**
	+ 如果升级`odo`和`blaze`包，请注意`networkx`用法更改部分
	+ 安装zipline之前，首先应安装`Bottleneck`和`statsmodels`
	+ 机器学习及优化包，务必在成功安装`zipline`之后才安装