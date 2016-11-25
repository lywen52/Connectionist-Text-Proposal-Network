import numpy as np
import tensorflow as tf
import cv2

import vgg16
import utils

img1 = utils.load_image("./Challenge2_Training_Task12_Images/101.jpg")
img2 = utils.load_image("./test_data/puzzle.jpeg")

batch1 = img1.reshape((1,img1.shape[0], img1.shape[1], img1.shape[2]))
#batch2 = img2.reshape((1, 1000, 1000, 3))

#batch = np.concatenate((batch1, batch2), 0)
batch = batch1
with tf.Session(
        config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    images = tf.placeholder("float", [1, img1.shape[0], img1.shape[1], img1.shape[2]])
    feed_dict = {images: batch}

    vgg = vgg16.Vgg16()
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    prob = sess.run(vgg.conv5_3, feed_dict=feed_dict)
    #print(prob)
    print prob.shape
    for i in range(0,1):
        
        prob[i] = prob[i]/np.amax(prob[i]);
        prob[i] = prob[i]*256
        temp = np.sum(prob[i], axis=2)
        cv2.imwrite("image" + str(i) + ".jpg",temp)
    #utils.print_prob(prob[0], './synset.txt')
    #utils.print_prob(prob[1], './synset.txt')
