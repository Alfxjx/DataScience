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

![np-week1](https://github.com/Alfxjx/DataScience/blob/master/pyDV/pic/np-week1.JPG)

###单元2

csv文件：存取一维、二维数组

~~~Python
np.savetxt(frame, array,fmt = '%18e',delimiter=None)#读取
np.loadtxt(frame, dtype=float,delimiter=None,unpack=false)#写入
#'%18e' 18位小数的浮点数
a = np.arange(12).reshape(3,4)
np.savetxt('a.csv',a,fmt='%d',delimiter=',')
~~~

~~~Python
.tofile(frame,sep=",",format='%s')#存
.fromfile(frame,dtype=float,count=-1,sep='')#取 
#sep为空，存取的是二进制
a = np.arange(100).reshape(5,10,2)
a.tofile("b.dat",sep=",",format='%d')
~~~

这种方法在存取过程需要知道原始数据的信息，需要新建一个元数据文件，存原始数据的维度信息。

~~~Python
np.save(frame, array)
np.load(frame, array)
np.savez(frame, array)# zip
#frame = .npy /.npz 格式
~~~

**随机数函数**

~~~Python
np.random.randn(d0,d1,....,dn) # 有n 表示正态分布
~~~

![seed](/seed.JPG)

_seed函数，每次生成相同的随机数组。_

~~~Python
>>> a = np.random.randint(100,200,(3,4))
>>> np.random.shuffle(a)
>>> np.random.permutation(a) #a 不变
>>> np.random.choice(a,size,replace=false,p=b/np.sum(b))
~~~

~~~python
uniform(low,high,size) 
#产生具有均匀分布的数组,low起始值,high结束值,size形状
normal(loc,scale,size)
#产生具有正态分布的数组,loc均值,scale标准差,size形状
poisson(lam,size)
#产生具有泊松分布的数组,lam随机事件发生率,size形状
~~~

**统计函数**

~~~python
std(a, axis=None) 
#根据给定轴axis计算数组a相关元素的标准差
var(a, axis=None)
#根据给定轴axis计算数组a相关元素的方差
~~~

**梯度函数**
~~~python
>>> a=np.random.randint(0,20,(5))
>>> a
array([18, 10, 18,  9, 16])
>>> np.gradient(a)
array([-8. ,  0. , -0.5, -1. ,  7. ])
>>> c= np.random.randint(0,50,(3,5))

>>> c #二维数组？？？？
array([[11, 44, 10, 38, 20],
       [46, 33, 19, 13, 33],
       [ 7, 32, 22, 19, 38]])
>>> np.gradient(c)
[array([[ 35. , -11. ,   9. , -25. ,  13. ],
       [ -2. ,  -6. ,   6. ,  -9.5,   9. ],
       [-39. ,  -1. ,   3. ,   6. ,   5. ]]), array([[ 33. ,  -0.5,  -3. ,   5. , -18. ],
       [-13. , -13.5, -10. ,   7. ,  20. ],
       [ 25. ,   7.5,  -6.5,   8. ,  19. ]])]
~~~
