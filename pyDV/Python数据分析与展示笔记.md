#Python数据分析与展示笔记
##第0周

~~~python
import numpy as np
a = np.arrange(10)
a?
#IPython环境中，可以用? 显示数据的具体信息。
~~~

~~~python
%run demo.py
#执行（在空的命名空间执行） demo.py
%reset 
%who
#查看与删除变量
~~~
##第一周
- 一维数据：列表、集合
- 二维数据：列表
- 高维数据：字典数据表示，json等

**Numpy** n维数组对象---ndarray

~~~python
import numpy as np
def npsum():
	a = np.array([0,1,2])
	b = np.array([8,7,6])
	npsum = a**2 + b**3
	return npsum
# 数组对应元素幂相加
~~~

- 轴axis（数据的维度）
- 秩rank（维度的数量）

**ndarray数组创建方法**

~~~python
x = np.array([1,2],(2,3))
allone = np.ones((2,3),dtype = int32)
allzeros = np.zeros(shape)
y = np.eye(n)
# 单位对角阵

x = np.arange(shape)
#顺序生成数组

np.ones_like(a) 
# 形状与a相同的全1数组

np.linespace(1,10,4,endpoint = false)
# 浮点数，从1到10， 总共四个，endpoint表示生成的数是否包括10

np.concatenate() 
#数组合并

.reshape(shape)#不改变原数组
.resize(shape)#改变
.astype(shape)#类型转变，必创建新数组
~~~

**数组的索引和切片**

