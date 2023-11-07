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
求解JCMsuite工程之前，首先应该进行求解器的设置。
JCMsolver求解器使用方法如下：
首先导入jcmutils包：
```python
import jcmutils
```
然后，初始化jcmutils.solver
```python
solver = jcmutils.solver(jcmp_path,database_path,keys)
```
初始化参数如下：
- jcmp_path: jcmsuite的工程文件的project.jcmp完整路径
- database_path: resultbag.db完整路径
- keys: 传给jcm模板的参数的列表。是\[key01,key02,key03\]这样的形式
初始化求解器后，`solver`变量中即包含了上述三个参数。随后，直接进行求解即可：
```python
solver.solve()
```
经过漫长的等待，即可求得结果。日志保存在上面初始化日志时提到的日志文件或控制台输出

## 查看结果
包内包含的功能可以查看jcmsuite的结果results，也可以查看图片、保存图片
使用方法如下：
首先应该已初始化并求解过`solver`。在已经求解过，`resultbag.db`中已包含了仿真结果的情况下，可以使用如下四个函数：
```python
#获取对应key的结果
solver.get_result(key)
#查看对应key的图像
solver.show_image(key,num_of_results,is_light_intense=False)
#保存对应key的图像
solver.save_image(target_directory,key,num_of_results,is_light_intense=False)
#保存所有图像
solver.save_all_image(target_directory,num_of_results,is_light_intense=False)
```
其中，各参数的含义如下：
- key: 想要查看或保存的对应key
- num_of_results: 想要查看或保存的图像在工程中的序号
- target_directory: 想要保存的图像的父目录
- is_light_intense: 当其为True时，保存光强而非电场强度

# 生成数据集
包内的函数可以生成对应的数据集，使用方法如下：
按照`dataset_utils.export_defect_datas()`函数的要求，提供对应的参数。随后，将每个缺陷图像所对应的参数返回的数据组成一个List，并提供给`dataset_utils.export_dataset()`函数。

示例：
```python
generator = jcmutils.datagen()
template_image = cv2.imread(template_image_path,cv2.IMREAD_GRAYSCALE)
list_datas = []
list_test_datas = []
for dir in list_defect_directory:
    list_imgs = os.listdir(dir)
    for imgs in list_imgs:
        image = cv2.imread(os.path.join(dir,imgs),cv2.IMREAD_GRAYSCALE)
        defect_class = 0 if "instruction" in dir else 1
        datas = generator.export_defect_datas(template_image,image,periodic_info,signal_level,defect_class)
        if datas[4]:
            list_datas.append(datas)
        else:
            list_test_datas.append(datas)

generator.export_dataset(list_datas,template_image,target_shape,source_density,target_density,os.path.join(dataset_dir,"train"),periodic_info,enhance_info,defect_num_one_image,min_required_num)

generator.export_dataset(list_test_datas,template_image,target_shape,source_density,target_density,os.path.join(dataset_dir,"test"),periodic_info,enhance_info,defect_num_one_image,min_required_num/10)
```