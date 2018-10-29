

```python
import numpy as np
import tensorflow as tf
```

    F:\code\anaconda\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    


```python
w = tf.Variable(0,dtype=tf.float32)
cost = tf.add(tf.add(w**2,tf.multiply(-10.0,w)),25.0)
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
```


```python
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
session.run(w)
session.run(train)
#cudaGetDevice() failed. Status: CUDA driver version is insufficient for CUDA runtime version
# need a cuda 8.0 for my computer
#see more in https://www.cnblogs.com/wolflzc/p/9117291.html
```


    -------------------------------------------------------------------

    InternalError                     Traceback (most recent call last)

    <ipython-input-5-91979d02e4c7> in <module>()
          1 init = tf.global_variables_initializer()
    ----> 2 session = tf.Session()
          3 session.run(init)
          4 session.run(w)
          5 session.run(train)
    

    F:\code\anaconda\lib\site-packages\tensorflow\python\client\session.py in __init__(self, target, graph, config)
       1509 
       1510     """
    -> 1511     super(Session, self).__init__(target, graph, config=config)
       1512     # NOTE(mrry): Create these on first `__enter__` to avoid a reference cycle.
       1513     self._default_graph_context_manager = None
    

    F:\code\anaconda\lib\site-packages\tensorflow\python\client\session.py in __init__(self, target, graph, config)
        632     try:
        633       # pylint: disable=protected-access
    --> 634       self._session = tf_session.TF_NewSessionRef(self._graph._c_graph, opts)
        635       # pylint: enable=protected-access
        636     finally:
    

    InternalError: cudaGetDevice() failed. Status: CUDA driver version is insufficient for CUDA runtime version



```python
print("W after one iteration:",session.run(w))
for i in range(1,1000):
    session.run(train)

print("W after 1000 iteration:", session.run(w))
```


    -------------------------------------------------------------------

    NameError                         Traceback (most recent call last)

    <ipython-input-6-f6ff42a41ce5> in <module>()
    ----> 1 print("W after one iteration:",session.run(w))
          2 for i in range(1,1000):
          3     session.run(train)
          4 
          5 print("W after 1000 iteration:", session.run(w))
    

    NameError: name 'session' is not defined

