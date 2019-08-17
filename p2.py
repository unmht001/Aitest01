# # import numpy as np
# # import tensorflow as tf

# # import matplotlib.pyplot as plt


# # learning_rate = 0.1
# # training_epochs = 3000
# # display_step = 50

# # train_x = np.asarray([3.0, 6.0, 9.0])
# # train_y = np.asarray([7.0, 9.0, 7.0])
# # X = tf.compat.v1.placeholder(tf.float32, shape=None, name="X")
# # Y = tf.compat.v1.placeholder(tf.float32, shape=None, name="Y")
# # cx = tf.Variable(3, name='cx', dtype=tf.float32)
# # cy = tf.Variable(3, name='cy', dtype=tf.float32)
# # distance = tf.pow(tf.add(tf.pow((X-cx), 2), tf.pow((Y-cy), 2)), 0.5)
# # mean = tf.reduce_mean(distance)
# # cost = tf.reduce_sum(tf.pow((distance-mean), 2)/3)

# # optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# # init = tf.compat.v1.global_variables_initializer()

# # with tf.compat.v1.Session() as sess:
# #     sess.run(init)

# #     for epoch in range(training_epochs):
# #         sess.run(optimizer, feed_dict={X: train_x, Y: train_y})
# #         c = sess.run(cost, feed_dict={X: train_x, Y: train_y})
# #         if (c-0) < 0.0000000001:
# #             break
# #         if (epoch+1) % display_step == 0:

# #             c = sess.run(cost, feed_dict={X: train_x, Y: train_y})
# #             m = sess.run(mean, feed_dict={X: train_x, Y: train_y})
# #             print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),
# #                   "CX=", sess.run(cx), "CY=", sess.run(cy), "Mean=", "{:.9f}".format(m))
# #     print("Optimization Finished!")
# #     training_cost = sess.run(cost, feed_dict={X: train_x, Y: train_y})
# #     print("Training cost=", training_cost, "CX=", round(sess.run(cx), 2),
# #           "CY=", round(sess.run(cy), 2), "R=", round(m, 2), '\n')

# # def sigmoid(x):
# #     return 1.0/(1+np.exp(-x))

# # sigmoid_inputs = np.arange(-10,10,0.1)
# # sigmoid_outputs = sigmoid(sigmoid_inputs)
# # print("Sigmoid Function Input :: {}".format(sigmoid_inputs))
# # print("Sigmoid Function Output :: {}".format(sigmoid_outputs))

# # plt.plot(sigmoid_inputs,sigmoid_outputs)
# # plt.xlabel("Sigmoid Inputs")
# # plt.ylabel("Sigmoid Outputs")
# # plt.show()

# from PIL import Image
# import matplotlib.pyplot as plt
# img = Image.open("D:\\fff.jpg")
# gray = img.convert("L")
# r, g, b = img.split()
# pic = Image.merge("RGB", (r, g, b))
# plt.figure("美女")
# plt.subplot(3, 3, 1), plt.title("origin")
# plt.imshow(img), plt.axis("off")
# plt.subplot(3, 3, 2), plt.title("gray")
# plt.imshow(gray, cmap="gray"), plt.axis("off")
# plt.subplot(3, 3, 3), plt.title("merge")
# plt.imshow(pic), plt.axis("off")
# plt.subplot(3, 3, 4), plt.title("r")
# plt.imshow(r, cmap="gray"), plt.axis("off")
# plt.subplot(3, 3, 5), plt.title("g")
# plt.imshow(g, cmap="gray"), plt.axis("off")
# plt.subplot(3, 3, 6), plt.title("b")
# plt.imshow(b, cmap="gray"), plt.axis("off")
# plt.subplot(3, 3, 7), plt.title("merge")
# plt.imshow(r), plt.axis("off")
# plt.subplot(3, 3, 8), plt.title("merge")
# plt.imshow(g), plt.axis("off")
# plt.subplot(3, 3, 9), plt.title("merge")
# plt.imshow(b), plt.axis("off")
# plt.show()

import math
import numpy as np


# class aa:
#     __mp = {}

#     def __getattr__(self, name):
#         if name[0] != '_':
#             return self.__mp.get(name, None)

#     def __setattr__(self, name, value):
#         if name[0] != '_':
#             self.__mp[name] = value

#     def __init__(self):
#         # self.__mp={}
#         self.init()

#     def init(self):
#         self.b11 = -0.74
#         self.b12 = -0.74
#         self.b21 = -102
#         self.w21 = 106
#         self.w22 = 106
#         self.h = -0.15

#     def s(self, num):

#         return 1/(1+np.exp(-num))

#     def train(self, lst):
#         for i in lst:
#             x1, x2, at = i
#             self.a11 = self.s(x1+self.b11)
#             self.a12 = self.s(x2+self.b12)
#             self.a21 = self.s(self.a11*self.w21+self.a12*self.w22+self.b21)
#             self.x1 = self.w21*self.a11*(1-self.a11)
#             self.x2 = self.w22*self.a12*(1-self.a12)
#             self.x3 = 1
#             self.x4 = self.a11
#             self.x5 = self.a12
#             self.x6 = self.h*(self.a21-at)/(self.a21*(1-self.a21)*2)
#             self.rr = self.x6 / \
#                 math.sqrt(self.x1**2 + self.x2**2 +
#                           self.x3**2 + self.x4**2+self.x5**2)

#             self.b11 += self.x1*self.rr
#             self.b12 += self.x2*self.rr
#             self.b21 += self.x3*self.rr
#             self.w21 += self.x4*self.rr
#             self.w22 += self.x5*self.rr
#             # print(0.5*(self.a21-at)**2,self.a21,self.cal(x1,x2))

#     def cal(self, x1, x2):
#         a11 = self.s(x1+self.b11)
#         a12 = self.s(x2+self.b12)
#         return self.s(self.w21*a11+self.w22*a12+self.b21)


# a = aa()

# for i in range(30000):
#     a.train([[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]])

# print("1 ~ 1 = {:.02f}\n0 ~ 0 = {:.02f}\n1 ~ 0 = {:.02f}\n0 ~ 1 = {:.02f}\nb11 is {}\nb12 is {}\nb21 is {}\nw21 is {}\nw22 is {}\n ".format(
#     a.cal(1, 1), a.cal(0, 0), a.cal(1, 0), a.cal(0, 1), a.b11, a.b12, a.b21, a.w21, a.w22))

class base:
    __mp = {}

    def __getattr__(self, name):
        if name[0] != '_':
            return self.__mp.get(name, None)

    def __setattr__(self, name, value):
        if name[0] != '_':
            self.__mp[name] = value

    # def __init__(self):
    #     # self.__mp={}
    #     self.init()
    
class Node(base):
    pass
