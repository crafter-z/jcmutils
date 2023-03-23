# JCMsuite自用简化模块
# 使用方法

## log系统
基于logging模块。
注意！！！如果想要使用本包，必须先初始化logger。
具体为：
在使用本包内的任何功能前，调用
```python
jcmutils.logger.init_logger(logger_name,use_logfile=False, logfile_path="jcmlog.log", log_format="|%(asctime)s - %(levelname)s|->%(message)s", data_format="%Y/%m/%d %H:%M:%S", log_level=logger_level.DEBUG)
```
其中参数：
- logger_name: 模块名称，等同于logging包中的logging.getlogger(logger_name)，随意取一个就行
- use_logfile: 控制台输出还是文件输出
- logfile_path: 如果使用文件输出，输出文件的路径
- log_format: 日志格式，如无必要不必修改
- data_format: 日志的日期格式，如无必要不必修改
- log_level:日志显示的等级，如无必要不必修改
## 生成科勒照明光
如果想要生成科勒照明光，则：
```python
import jcmutils
keys = jcmutils.gen_kholer_sources(maxtheta, phi0, spacing, lambda0, flag_is_symmetry=False)
```
其中，函数的参数分别为：
- maxtheta:科勒照明中的最大照明角
- phi0:科勒照明光瞳面上偏振方向与与x轴之间的夹角
- spacing:在科勒照明光瞳面上的取样间隔
- lambda0:波长
- flag_is_symmetry:是否采用对称简化计算。如果为真，则只生成一半的光源

## 解JCMsuite工程
