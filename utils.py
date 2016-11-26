import skimage
import skimage.io
import skimage.transform
import numpy as np
import vgg16
import tensorflow as tf

class box:
    def __init__(self, data):
        self.left=int(data[0])
        self.top=int(data[1])
        self.right=int(data[2])
        self.bottom=int(data[3])
        self.word=data[4]
    
class image_data: 
    def __init__(self, path):
        self.rectangles=[]

        self.path = path
        self.debug = True
        self.features = None
        self.orignal_image = self.load_image(path)
        self.image=self.resize_image(self.orignal_image)

        self.scale = float(self.image.shape[0])/float(self.orignal_image.shape[0])

        if self.debug:
            print "Oringal Image " , self.orignal_image.shape
            print "Resized Image" , self.image.shape
            
            print "Scale = " , self.scale

    def resize_image(self, img):
        if(img.shape[0] < img.shape[1]):
            image = skimage.transform.resize(img, (600, int(600.0/img.shape[0]*float(img.shape[1])),3))
        else:
            image = skimage.transform.resize(img, (int(600.0/img.shape[1]*img.shape[0]),600,3))
        return image

    
    def load_features(self):
        """
        Loads features of the image from the same path as image. Features must be pre-computed by calling the compute_features() function on the directory


        Side Affects
        ------------
        Features are loaded in self.features. 
        """
        import os.path
        if os.path.isfile(self.path+".npy"):
            self.features = np.load(self.path+".npy")
            if self.debug:
                print "Features Loaded. Shape = ", self.features.shape
            else:
                print "Please compute features of the images first using compute_features() before running this"
            
            
    def load_gt(self):
        """
        First renamed gt files from gt_100.txt to 100.jpg.gt. 
        Used rename -v -n 's/gt_//' *.txt 
        Followed by rename -v -n 's/.txt/.jpg.gt/' *.txt

        Then use this function to store GTs into a list of boxes

        GT Format :each line represents : left top right bottom "Text" (ICDAR 2015 localization challegent format) 
        """
        self.lines = tuple(open(self.path+".gt", 'r'))
        for line in self.lines:
            words = line.split()
            self.rectangles.append(box(words))
        if self.debug:
            print len(self.rectangles)
                
        



    def load_image(self,path):

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
                    img = image_data(path+"/"+file).image
                    img = np.expand_dims(img,axis=0)
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

    #compute_features("./Train_Images")
    myImage = image_data("./Train_Images/102.jpg")
    myImage.load_features()
    myImage.load_gt()

if __name__ == "__main__":
    test()
