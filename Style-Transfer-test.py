
# 测试函数前后位置是否有影响
def add(a, b):
    c = multiply(a, b)
    return c + 1

def multiply(a, b):
    return a * b

# 结论：不影响
import numpy as np
import os


# 保存变量
import tensorflow as tf

# 创建两个变量
v1 = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="v_1")
v2 = tf.Variable(tf.zeros([200]), name="v_2")

# 添加用于初始化变量的节点
init_op = tf.global_variables_initializer()

# Create a saver.
saver = tf.train.Saver(tf.global_variables())


# 运行，保存变量
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for step in range(5000):
        sess.run(init_op)
        if step % 1000 == 0:
            saver.save(sess, 'modelTest/' + 'my-model', global_step=step)


from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
print_tensors_in_checkpoint_file('modelTest/' + "my-model-0", None, True)  # 通过这个方法，我们可以打印出保存了什么变量和值。

saver = tf.train.Saver()
with tf.Session() as sess:
    #tf.global_variables_initializer().run()
    module_file = tf.train.latest_checkpoint( 'modelTest')
    saver.restore(sess, module_file)
    #print("w1:", sess.run(v3))
    #print("b1:", sess.run(v4))
    # print("w1:", sess.run(v1))
    # print("b1:", sess.run(v2))


# if __name__ == '__main__':
#     a = os.path.join('model', 'fnk.ckpt')
#     model = os.path.dirname(a) + '/'
#     print(model)

if __name__ == '__main__':
    print(int(7 / 4))