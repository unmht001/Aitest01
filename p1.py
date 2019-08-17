import math
import warnings

import cv2
import imageio
import matplotlib.cbook
import matplotlib.pyplot as plt
import numpy as np
import pylab
# import math
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import convolve2d
from skimage import color, exposure, measure
from sklearn.cluster import KMeans

import tensorflow as tf

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
pic = imageio.imread('aa1.png')
fig = plt.figure(figsize=(6, 6))


def f1(pic):
    plt.imshow(pic)
    pylab.show()
    plt.axis('off')


def f2(pic):
    neg = 255-pic

    plt.imshow(neg)
    pylab.show()
    plt.axis('off')


def f3(pic):
    def gray(rgb): return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    gray = gray(pic)
    max_ = np.max(gray)

    def log_transform():
        return (255/np.log(1+max_)) * np.log(1+gray)
    return log_transform()


def f4(pic):
    gamma = 2.2  # Gamma < 1 ~ Dark  ;  Gamma > 1 ~ Bright
    gamma_correction = ((pic/255) ** (1/gamma))
    plt.imshow(gamma_correction)
    pylab.show()
    plt.axis('off')


def f5(pic):
    def Convolution(image, kernel):
        conv_bucket = []
        for d in range(image.ndim):
            conv_channel = convolve2d(
                image[:, :, d], kernel, mode="same", boundary="symm")
            conv_bucket.append(conv_channel)
        return np.stack(conv_bucket, axis=2).astype("uint8")
    kernel_sizes = [9, 15, 30, 60]
    fig, axs = plt.subplots(nrows=1, ncols=len(kernel_sizes), figsize=(15, 15))
    for k, ax in zip(kernel_sizes, axs):
        kernel = np.ones((k, k))
        kernel /= np.sum(kernel)
        ax.imshow(Convolution(pic, kernel))
        ax.set_title("Convolved By Kernel: {}".format(k))
        ax.set_axis_off()
    pylab.show()


def f6(pic):
    def gaussain_filter_(img):
        """
        Applies a median filer to all channels
        """
        ims = []
        for d in range(3):
            img_conv_d = gaussian_filter(img[:, :, d], sigma=4)
            ims.append(img_conv_d)
        return np.stack(ims, axis=2).astype("uint8")
    filtered_img = gaussain_filter_(pic)

    sobel_x = np.c_[
        [-1, 0, 1],
        [-5, 0, 5],
        [-1, 0, 1]
    ]
    # top sobel
    sobel_y = np.c_[
        [1, 5, 1],
        [0, 0, 0],
        [-1, -5, -1]
    ]
    ims = []
    for i in range(3):
        sx = convolve2d(filtered_img[:, :, i],
                        sobel_x, mode="same", boundary="symm")
        sy = convolve2d(filtered_img[:, :, i],
                        sobel_y, mode="same", boundary="symm")
        ims.append(np.sqrt(sx*sx + sy*sy))
    img_conv = np.stack(ims, axis=2).astype("uint8")
    return img_conv


def f7(pic):
    pic2 = pic[:]
    h, w, f = pic2.shape

    im_small_long = pic2.reshape((h * w, f))
    im_small_wide = im_small_long.reshape((h, w, f))
    km = KMeans(n_clusters=2)
    km.fit(im_small_long)
    seg = np.asarray([(1 if i == 1 else 0)for i in km.labels_]).reshape((h, w))
    contours = measure.find_contours(seg, 0.5, fully_connected="high")
    simplified_contours = [measure.approximate_polygon(
        c, tolerance=5) for c in contours]
    plt.figure(figsize=(5, 10))
    for n, contour in enumerate(simplified_contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.ylim(h, 0)
    plt.axes().set_aspect('equal')
    pylab.show()


def f8(pic):
    img = color.rgb2gray(pic)
    kernel = np.array(
        [[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]])
    kernel = kernel / np.sum(kernel)
    edges = convolve2d(img, kernel, mode='same')
    edges_equalized = exposure.equalize_adapthist(
        edges/np.max(np.abs(edges)), clip_limit=0.03)
    return edges_equalized


def f9(pic):
    data = (50, 500)
    img = cv2.Canny(pic, *data)
    return img


def f10(pic):

    def beta_pdf(x, a, b):
        return (x**(a-1) * (1-x)**(b-1) * math.gamma(a + b)
                / (math.gamma(a) * math.gamma(b)))

    class UpdateDist(object):
        def __init__(self, ax, prob=0.5):
            self.success = 0
            self.prob = prob
            self.line, = ax.plot([], [], 'k-')
            self.x = np.linspace(0, 1, 200)
            self.ax = ax

            # Set up plot parameters
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 15)
            self.ax.grid(True)

            # This vertical line represents the theoretical value, to
            # which the plotted distribution should converge.
            self.ax.axvline(prob, linestyle='--', color='black')

        def init(self):
            self.success = 0
            self.line.set_data([], [])
            return self.line,

        def __call__(self, i):
            # This way the plot can continuously run and we just keep
            # watching new realizations of the process
            if i == 0:
                return self.init()

            # Choose success based on exceed a threshold with a uniform pick
            if np.random.rand(1,) < self.prob:
                self.success += 1
            y = beta_pdf(self.x, self.success + 1, (i - self.success) + 1)
            self.line.set_data(self.x, y)
            return self.line,

    # Fixing random state for reproducibility
    np.random.seed(56213546)

    fig, ax = plt.subplots()
    ud = UpdateDist(ax, prob=0.7)
    anim = FuncAnimation(fig, ud, frames=np.arange(
        100), init_func=ud.init, interval=100, blit=True)
    plt.show()


def f11(pic):
    # def _ft(pic):
    #     ims = []
    #     for d in range(3):
    #         img_conv_d = gaussian_filter(pic[:, :, d], sigma=2)
    #         # img_conv_d = convolve2d(img_conv_d,np.array([[-1/8,-1/8,-1/8],[-1/8,2,-1/8],[-1/8,-1/8,-1/8]]),mode='same')
    #         ims.append(img_conv_d)
    #     img= np.stack(ims, axis=2).astype("uint8")
    #     return img
    # img=pic[:,:,:]
    # data = (0, 0)
    # plt.subplot(1,2,1)

    # plt.imshow(img)
    # a=plt.text(0,0,str(data))
    # aaa=[img]
    # def ud(i):
    #     data=(i,i)
    #     aaa[0] = _ft(aaa[0])

    #     a._text=str(data)
    #     return plt.imshow(f3(aaa[0]))
    # ani=FuncAnimation(fig,ud,frames=range(128),interval=1000)
    # pylab.show()
    img = color.rgb2gray(pic)

    def ff1(x, a, b):
        if a < x <= b:
            return (x-a)/(b-a)*255
        else:
            return 0
    ff1 = np.vectorize(ff1)
    r = ff1(img, 0, 0.299)
    g = ff1(img, 0.299, 0.299+0.587)
    b = ff1(img, 0.299+0.587, 1)
    img = np.stack([r, g, b], axis=2).astype("uint8")

    plt.imshow(img, cmap='gray')
    plt.show()

# f11(pic)


x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3

#create tensorflow structure start###
Weights = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data+biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.compat.v1.global_variables_initializer()
#create tensorflow structure end###

sess = tf.compat.v1.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
    