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
    print img.shape
    if(img.shape[0] < img.shape[1]):
        image = skimage.transform.resize(img, (600, int(600.0/img.shape[0]*float(img.shape[1])),3))
    else:
        image = skimage.transform.resize(img, (int(600.0/img.shape[1]*img.shape[0]),600,3))
    batch = np.expand_dims(image,axis=0)
    print batch.shape
    return batch


def compute_features(path, network = "VGG16", extension = "jpg"):
    
    """
    Computes VGG16 last conv layer features of images in a directory and save the result in npy files with same names.

    Extended description of function.

    Parameters
    ----------
    path : str
        Path name of the folder containing the files
    network : str
        For extending to other networks
    extension : str
        Format of image in the directory. All other formats will be ignored. 
    Returns
    -------
    void
        Nothing 

    Side Affects
    ------------
    Files saved in the directory containing the images
    """
    if(network == "VGG16"):
        
        
        with tf.Session(
                config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.9)))) as sess:

            
            images = tf.placeholder("float", shape = None)

            vgg = vgg16.Vgg16()
            with tf.name_scope("content_vgg"):
                vgg.build(images)

            import os
            for file in os.listdir(path):
                if file.endswith(extension):
                    print(file)
                    img =  load_image(path+"/"+file)
                    print img.shape
                    feed_dict = {images: img}

                    features = sess.run(vgg.conv5_3, feed_dict=feed_dict)
                    print features[0].shape
                    np.save(path+"/"+file, features[0])

    else:
        print ("Implement functionality of your desired model")

def test():
    print "Testing"

    compute_features("./Challenge2_Training_Task12_Images")

if __name__ == "__main__":
    test()
