import utils 

data = utils.load_images("./Train_Images")
print len(data)

print "Data Loaded"


def sliding_window(ImageObject, mode='edge'):
    import skimage

    array =ImageObject.features
    array_padded = np.pad(data[0].features,((1,1),(1,1),(0,0)), mode=mode)
    out = skimage.util.view_as_windows(array_padded, (3,3,512), step=1)
    ImageObject.rolling_windows=out
    return out
    #Older implementation 
    # array_padded = np.pad(array,((1,1),(1,1),(0,0)), mode=mode)
    # for y in range(1,array_padded.shape[1]-1):
    #     for x in range(1,array_padded.shape[0]-1):
    #         col= array_padded[x-1:x+2, y-1:y+2]

print data[0].rolling_windows.shape
