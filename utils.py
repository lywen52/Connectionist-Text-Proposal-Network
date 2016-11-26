import skimage
import skimage.io
import skimage.transform
import numpy as np
import vgg16
import tensorflow as tf


# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]

class image_data: 
    def __init__(self, path):

        self.orignal_image = self.load_image(path)
        self.image=self.resize_image(self.orignal_image)
        self.data = self.expand_dims(self.image)
        self.scale = float(self.image.shape[0])/float(self.orignal_image.shape[0])

        print "Oringal Image " , self.orignal_  bimage.shape
        print "Resized Image" , self.image.shape

        print "Scale = " , self.scale

    def resize_image(self, img):
        if(img.shape[0] < img.shape[1]):
            image = skimage.transform.resize(img, (600, int(600.0/img.shape[0]*float(img.shape[1])),3))
        else:
            image = skimage.transform.resize(img, (int(600.0/img.shape[1]*img.shape[0]),600,3))
        return image

    def expand_dims(self,img):
        return np.expand_dims(img,axis=0)


    def load_image(self,path):
        # load image
        img = skimage.io.imread(path)
        img = img / 255.0
        assert (0 <= img).all() and (img <= 1.0).all()

        return img



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
                    img = image_data(path+"/"+file).data
                    #img =  load_image(path+"/"+file)
                    print img.shape
                    feed_dict = {images: img}

                    features = sess.run(vgg.conv5_3, feed_dict=feed_dict)
                    print features[0].shape
                    #np.save(path+"/"+file, features[0])

    else:
        print ("Implement functionality of your desired model")

def test():
    print "Testing"

    compute_features("./Train_Images")

if __name__ == "__main__":
    test()
