# JCMsuite自用简化模块
# 使用方法
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
- flag_is_symmetry:是否采用