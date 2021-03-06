import skimage
import skimage.io
import skimage.transform
import numpy as np
import vgg16
import tensorflow as tf

class box:
    """
    Class to store the SCALED bounding boxes information
    """
    def __init__(self, data, scale=1):
       
        self.left=int(int(data[0])*scale)
        self.top=int(int(data[1])*scale)
        self.right=int(int(data[2])*scale)
        self.bottom=int(int(data[3])*scale)
        self.height= abs(self.top-self.bottom)
        self.width = abs(self.right - self.left)

        self.word=data[4]

    def draw_box(self,img):
        """Draw the rectangle on the image passed to the function"""
        import cv2
        img = myImage.image
        img = img*256
        cv2.rectangle(img,(self.left,self.top),(self.right,self.bottom),(0,255,0),3)
        return img

    def __str__(self):
        """For printing a Box attributes"""
        
        return "Left: "+ str(self.left)+ " Top: "+ str(self.top)+" Right: "+ str(self.right)+ " Bottom: "+ str(self.bottom) + " Word = "+ str(self.word)
class image_data: 
    def __init__(self, path):
        self.rectangles=[]
        self.rolling_windows=None
        self.path = path
        self.debug = False
        self.features = None
        orignal_image = self.load_image(path)
        self.image=self.resize_image(orignal_image)

        self.scale = float(self.image.shape[0])/float(orignal_image.shape[0])

        if self.debug:
            print "Oringal Image " , orignal_image.shape
            print "Resized Image" , self.image.shape
            
            print "Scale = " , self.scale
        self.image=None

    def resize_image(self, img):
        if(img.shape[0] < img.shape[1]):
            image = skimage.transform.resize(img, (600, int(600.0/img.shape[0]*float(img.shape[1])),3))
        else:
            image = skimage.transform.resize(img, (int(600.0/img.shape[1]*img.shape[0]),600,3))
        return image

    def load_scaled_image(self):
        """
        Returns the resized image from the HARDDRIVE. Slow inside a loop
        """
        return self.resize_image(self.load_image())
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
            self.rectangles.append(box(words,self.scale))
        if self.debug:
            print "Loaded GT"
            for r in self.rectangles:
                print r
                
        

    

    def load_image(self):

        img = skimage.io.imread(self.path)
        img = img / 255.0
        assert (0 <= img).all() and (img <= 1.0).all()

        return img

    def sliding_window(self, mode='edge'):
        import skimage

        array =self.features
        array_padded = np.pad(array,((1,1),(1,1),(0,0)), mode=mode)
        out = skimage.util.view_as_windows(array_padded, (3,3,512), step=1)
        self.rolling_windows=out
        return out



def load_images(path, extension = "jpg"):
        data = []
        import os
        for file in os.listdir(path):
            if file.endswith(extension):
                print(file)
                img = image_data(path+"/"+file)
                img.load_gt()
                img.load_features()
                img.sliding_window()
                data.append(img)
        return data

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
                    np.save(path+"/"+file, features[0])

    else:
        print ("Implement functionality of your desired model")

def test():
    print "Testing"

    #compute_features("./Train_Images")
    myImage = image_data("./Train_Images/102.jpg")
    numpy_data = np.load("./Train_Images/102.jpg.npy")
    print numpy_data.shape
    myImage.load_features()
    myImage.load_gt()

    import cv2
    img = myImage.image
    img = img*256
    for rec in myImage.rectangles:
        cv2.rectangle(img,(rec.left,rec.top),(rec.right,rec.bottom),(0,255,0),3)
    
    cv2.imwrite("result.jpg",img)






if __name__ == "__main__":
    test()
