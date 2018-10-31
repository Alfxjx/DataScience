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
~~~python
#索引
a = np.arange(24).reshape((2,3,4))
a
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
a[1,2,3] #23
#第一个维度的第1个，第二维度的第2个，第三维度的第三个
#从0开始

a[-1,-2,-3]#17
#负数从右向左，-1就是右边第一个

#切片
a[:,1,-3] #array([ 5, 17])
#不考虑第一维度，只考虑第二维度是1，和第三维度是-3的。
a[:,:,::2] # 第三维度步长跳跃切片
array([[[ 0,  2],
        [ 4,  6],
        [ 8, 10]],

       [[12, 14],
        [16, 18],
        [20, 22]]])
~~~

**ndarray数组运算**
~~~python
#数组与标量计算
a = np.arange(24).reshape((2,3,4))
a.mean() #平均值
a = a/a.mean()
a
array([[[0.        , 0.08695652, 0.17391304, 0.26086957],
        [0.34782609, 0.43478261, 0.52173913, 0.60869565],
        [0.69565217, 0.7826087 , 0.86956522, 0.95652174]],

       [[1.04347826, 1.13043478, 1.2173913 , 1.30434783],
        [1.39130435, 1.47826087, 1.56521739, 1.65217391],
        [1.73913043, 1.82608696, 1.91304348, 2.        ]]])
~~~
_注意数组是否真实改变_
numpy 函数是对对应元素进行计算
**总结**

![np-week1](/np-week1.JPG)
