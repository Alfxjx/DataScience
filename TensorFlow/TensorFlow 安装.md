#TensorFlow 安装#
##准备工作
首先需要看一下，你的电脑是否有NVIDIA的显卡，是否支持CUDA。
###查看电脑显卡的驱动。
我的电脑是GTX755M，驱动版本375.61
查看驱动版本的方法见[这个链接](https://jingyan.baidu.com/article/eae078276932761fec5485b7.html).
根据[这里](http://www.cnblogs.com/LearnFromNow/p/9417272.html)的表，判断一下该下哪个版本的CUDA。

![对应的版本](https://github.com/Alfxjx/DataScience/blob/master/TensorFlow/png/cuda%26DriverVersion.png)

###对于本机，应该下载的是CUDA 8.0版本。
下载[CUDA8.0](https://developer.nvidia.com/cuda-downloads)
这里注意，对应的CUDA 8.0版本需要下载cuDNN的版本为6.0，以及这样的配置下，需要安装的TensorFlow-gpu的版本为1.4.0。下面是下载链接。
[Tensorflow1.4.0](https://pypi.org/project/tensorflow/1.4.0/)

###cuDNN的安装方法。
下载[cuDNN 6.0](https://developer.nvidia.com/cudnn)
解压之后，将对应的文件夹里面的文件，复制到上文中安装的CUDA的对应文件夹下。
默认的CUDA目录:  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0  
###验证是否安装成功
全部安装完之后，可以用[这个链接](https://github.com/Alfxjx/DataScience/tree/master/TensorFlow)里面的DL_tf_test.py

`import numpy as np`
`import tensorflow as tf`

`w = tf.Variable(0,dtype=tf.float32)`
`cost = tf.add(tf.add(w**2,tf.multiply(-10.0,w)),25.0)`
`train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)`

`init = tf.global_variables_initializer()`
`session = tf.Session()`
`session.run(init)`
`session.run(w)`
`session.run(train)`

`print("W after one iteration:",session.run(w))`
`for i in range(1,1000):`
    `session.run(train)`

`print("W after 1000 iteration:", session.run(w))`

执行结果如下：

`F:\code\c\DataScience\TensorFlow>python DL_tf_test.py`
`W after one iteration: 0.099999994`
`W after 1000 iteration: 4.999988`

这样就完成了TensorFlow的安装。
