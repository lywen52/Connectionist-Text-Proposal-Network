import skimage
import skimage.io
import skimage.transform
import numpy as np
import vgg16
import tensorflow as tf


# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    return img


def compute_features(path, network = "VGG16", extension=".jpg"):
    
    img1 = load_image(path) 
    print "Shape = " , img1.shape
    if(network == "VGG16"):
        batch = np.expand_dims(img1,axis=0)

        with tf.Session(
                config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
            images = tf.placeholder("float", [1, img1.shape[0], img1.shape[1], img1.shape[2]])
            feed_dict = {images: batch}

            vgg = vgg16.Vgg16()
            with tf.name_scope("content_vgg"):
                vgg.build(images)

            features = sess.run(vgg.conv5_3, feed_dict=feed_dict)
            print features[0].shape
            np.save(path, features[0])
    else:
        print ("Implement functionality of your desired model")

def test():
    print "Testing"
    compute_features("./Challenge2_Training_Task12_Images/101.jpg")

if __name__ == "__main__":
    test()
