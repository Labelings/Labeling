# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from datetime import datetime

import numpy as np
from tifffile import imread

import MetaSegmentMerger as msm

result = np.zeros((512, 512))
resolution = (512, 512)


def read_img():
    files = []
    for i in range(1, 6):
        for j in range(1, 5):
            files.append("C:/mpicbg/metaseg_data/Euclidean_squared_distance/000/diverse" + str(i) + "/sweep_" + str(
                j) + "_mask.tif")
    images = imread(files)
    return images


if __name__ == '__main__':

    # init
    images = read_img()
    patch_size = 64

    start = datetime.now()
    #initialize the merger
    merger = msm.MetaSegmentMerger.fromValues(first_image=images[0])
    # iterate over all images
    for image in images[1:]:
        #start position
        x, y = 0, 0
        #create 64x64 patches from each image
        patches = np.vsplit(image, int(image.shape[0] / patch_size))
        patches = [np.hsplit(seg, int(image.shape[1] / patch_size)) for seg in patches]
        for patchList in patches:
            for patch in patchList:
                # print(x,y)
                merger.add_segments(patch, x, y)
                y += patch_size
            y = 0
            x += patch_size
    merger.save_result("iterate_segments1")
    print(datetime.now() - start)

    start = datetime.now()
    merger2 = msm.MetaSegmentMerger.fromValues(first_image=images[0])
    merger2.iterate_over_images(images[1:])
    merger2.save_result("iterate_images1")
    print(datetime.now() - start)

