# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from datetime import datetime

import numpy as np
from tifffile import imread

import Labeling as lb
from line_profiler import LineProfiler

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


def test1():
    # init
    images = read_img()
    patch_size = 64
    profiler = LineProfiler()
    start = datetime.now()
    # initialize the merger
    merger = lb.Labeling.fromValues(first_image=images[0])
    # profiler.add_function(lb.Labeling.add_segments)
    # profiler.add_function(lb.Labeling.iterate_over_images)
    # iterate over all images
    for image in images[1:]:
        # start position
        x, y = 0, 0
        # create 64x64 patches from each image
        patches = np.vsplit(image, int(image.shape[0] / patch_size))
        patches = [np.hsplit(seg, int(image.shape[1] / patch_size)) for seg in patches]
        for patchList in patches:
            for patch in patchList:
                # print(x,y)
                profiler.runcall(merger.add_segments, patch, (x,y))
                #merger.add_segments(patch=patch, position=(x, y))
                y += patch_size
            y = 0
            x += patch_size
    #merger.save_result("iterate_segments1")
    profiler.print_stats()
    profiler.dump_stats("profiling.lprof")
    # print(datetime.now() - start)
    # start = datetime.now()
    # merger2 = lb.Labeling.fromValues(first_image=images[0])
    # merger2.iterate_over_images(images[1:])
    # merger2.save_result("iterate_images1")
    # print(datetime.now() - start)



def test2():
    global result, img
    a = np.zeros((4, 4), np.int32)
    a[:2] = 1
    example1_images = []
    example1_images.append(a)
    b = a.copy()
    b[:2] = 2
    example1_images.append(np.flip(b.transpose()))
    c = a.copy()
    c[:2] = 3
    example1_images.append(np.flip(c))
    d = a.copy()
    d[:2] = 4
    example1_images.append(d.transpose())
    # Initialize the merger with the first image. This can also be an empty image of zeros in the correct shape
    merger = lb.Labeling.fromValues(first_image=example1_images[0])
    patch_size = (4, 2)
    for image in example1_images[1:]:
        # start position
        x, y = 0, 0
        # create 2x2 patches from each image
        patches = np.vsplit(image, int(image.shape[0] / patch_size[0]))
        patches = [np.hsplit(seg, int(image.shape[1] / patch_size[1])) for seg in patches]
        for patchList in patches:
            for patch in patchList:
                # add a patch at the defined spot
                result = merger.add_segments(patch, (x, y))
                print(result)
                y += patch_size[1]
            y = 0

            x += patch_size[0]
        patch_size = (patch_size[1], patch_size[0])
    img, labeling = merger.save_result("example1")
    print(img)
    print(vars(labeling))
    # add_anything(data, (x,y,z):Tuple, merge:dict=None)
    # returns dict of actual labels dif


def test4():
    global result, img
    a = np.zeros((4, 4), np.int32)
    a[:2] = 1
    example1_images = []
    example1_images.append(a)
    b = a.copy()
    example1_images.append(np.flip(b.transpose()))
    c = a.copy()
    example1_images.append(np.flip(c))
    d = a.copy()
    example1_images.append(d.transpose())
    # Initialize the merger with the first image. This can also be an empty image of zeros in the correct shape
    merger = lb.Labeling.fromValues(first_image=example1_images[0])
    patch_size = (4, 2)
    for image in example1_images[1:]:
        # start position
        x, y = 0, 0
        # create 2x2 patches from each image
        patches = np.vsplit(image, int(image.shape[0] / patch_size[0]))
        patches = [np.hsplit(seg, int(image.shape[1] / patch_size[1])) for seg in patches]
        for patchList in patches:
            for patch in patchList:
                # add a patch at the defined spot
                result = merger.add_segments(patch, (x, y))
                print(result)
                y += patch_size[1]
            y = 0

            x += patch_size[0]
        patch_size = (patch_size[1], patch_size[0])
    img, labeling = merger.save_result("example1")
    print(img)
    print(vars(labeling))
    # add_anything(data, (x,y,z):Tuple, merge:dict=None)
    # returns dict of actual labels dif



def test3():
    # init
    images = read_img()
    patch_size = 64
    # profiler = LineProfiler()
    start = datetime.now()
    # initialize the merger
    merger = lb.Labeling.fromValues(first_image=images[0])
    # profiler.add_function(lb.Labeling.add_segments)
    # profiler.add_function(lb.Labeling.iterate_over_images)
    # iterate over all images
    for image in images[1:]:
        # start position
        x, y = 0, 0
        # create 64x64 patches from each image
        patches = np.vsplit(image, int(image.shape[0] / patch_size))
        patches = [np.hsplit(seg, int(image.shape[1] / patch_size)) for seg in patches]
        for patchList in patches:
            for patch in patchList:
                # print(x,y)
                # profiler.runcall(merger.add_segments, patch, (x,y))
                merger.add_segments(patch=patch, position=(x, y))
                y += patch_size
            y = 0
            x += patch_size
    # merger.save_result("iterate_segments1")
    print(datetime.now() - start)
    # profiler.print_stats()
    # profiler.dump_stats("profiling.lprof")


if __name__ == '__main__':

    test2()
    test4()
