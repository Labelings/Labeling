import copy
import os
from typing import List, Callable, Tuple, T

import numpy as np
from PIL import Image
from tifffile import imread

import labeling.bsoncontainer as bc


class LabelSet:
    set: set()
    source: str()

    def __init__(self, source="", element=None):
        self.source = source
        self.set = set()
        if element is not None:
            self.set.add(element)

    def add(self, element):
        self.set.add(element)


class Labeling:

    def __init__(self, shape: (int, int) = (512, 512), type: object = np.int8):
        self.result_image = np.zeros(shape, type)
        self.image_resolution = shape
        self.label_value = 0
        self.pixel_value = 1
        self.label_sets = {"0": set()}
        self.list_of_unique_ids = []
        self.unique_map_id_id_label = {}
        self.segmentation_source = {}
        self.metadata = None
        self.img_filename = None

    @classmethod
    def from_file(cls, path: str):
        labeling = cls.__new__(cls)
        container = bc.BsonContainer.decode(path)
        labeling.result_image = container.get_image()
        labeling.image_resolution = labeling.result_image.shape
        labeling.label_sets = container.labelSets
        labeling.metadata = container.metadata
        labeling.img_filename = os.path.split(path)[1].replace(".bson", ".tif")
        labeling.segmentation_source = dict.fromkeys(range(container.numSources), range(container.numSources))
        return labeling

    # @classmethod
    # def from_file_withfunc(cls, path: str, func: Callable[[int], T]):
    #     cls.labels = bc.BsonContainer.decode_withfunc(path, func)
    #     cls.result_image = cls.labels.get_image()
    #     cls.img = cls.labels.get_image()
    #     return cls

    @staticmethod
    def read_images(file_paths: list):
        return imread(file_paths)

    @classmethod
    def fromValues(cls, first_image: np.ndarray = np.zeros((512, 512), np.int8), source_id=None):
        labeling = Labeling()
        labeling.list_of_unique_ids = np.unique(first_image)
        labeling.image_resolution = first_image.shape
        labeling.result_image = np.zeros(first_image.shape, first_image.dtype)
        labeling.add_image(first_image,source_id)
        return labeling

    def iterate_over_images(self, images: List[np.ndarray], source_ids=None):
        # iterate over all images
        for image, source_id in zip(images, source_ids):
            self.add_image(image, source_id)

    def add_image(self, image: np.ndarray, source_id=None):
        self.add_segments(image, (0, 0), source_id=source_id)

    def add_segments(self, patch: np.ndarray, position: Tuple, merge: dict = None, source_id=None):
        list_of_unique_labelvalues = []
        segment_mapping = {}
        unique_map_id_id_label = {}
        temp = np.reshape(self.result_image, self.image_resolution)
        with np.nditer(patch, flags=["multi_index"], op_flags=["readonly"]) as it:
            for val in it:
                if val.item() != 0:
                    v = val.item()
                    pos = tuple(sum(x) for x in zip(it.multi_index, position))
                    if (temp[pos], v) not in unique_map_id_id_label.keys():

                        if v not in list_of_unique_labelvalues:
                            self.label_value += 1
                            list_of_unique_labelvalues.append(v)
                        unique_map_id_id_label[(temp[pos], v)] = (self.pixel_value, self.label_value)
                        if temp[pos] == 0:
                            self.label_sets[str(self.pixel_value)] = set()
                            self.label_sets[str(self.pixel_value)].add(self.label_value)
                            self.add_segmentation_source(source_id, self.label_value)
                            segment_mapping[str(self.pixel_value)] = LabelSet(source_id, self.label_value)
                        else:
                            labelset = copy.deepcopy(self.label_sets[str(temp[pos])])
                            labelset.add(self.label_value)
                            if str(self.pixel_value) not in self.label_sets.keys():
                                self.label_sets[str(self.pixel_value)] = labelset
                            self.add_segmentation_source(source_id, self.label_value)

                        temp[pos] = self.pixel_value
                        self.pixel_value += 1

                    else:
                        temp[pos] = unique_map_id_id_label[(temp[pos], v)][0]

        self.result_image = temp.flatten()
        return segment_mapping

    def add_metadata(self, data):
        self.metadata = data

    def add_segmentation_source(self, source_id, label_value):
        if source_id is not None:
            if str(source_id) not in self.segmentation_source.keys():
                self.segmentation_source[str(source_id)] = set()
            self.segmentation_source[str(source_id)].add(label_value)

    def save_result(self, path: str, cleanup: bool = True, save_json: bool = False):
        if cleanup:
            self.cleanup_labelsets()
        img = Image.fromarray(np.reshape(self.result_image, self.image_resolution))
        img.save(path + '.tif', 'tiff')
        self.img_filename = os.path.splitext(os.path.basename(path))[0] + '.tif'
        bson_con = bc.BsonContainer.fromValues(2, len(self.label_sets), len(self.segmentation_source),
                                               os.path.splitext(os.path.basename(path))[0] + '.tif', {},
                                               self.label_sets, self.metadata)
        bson_con.encode_and_save(path + '.bson')
        # optional, just to easily content check
        if save_json:
            bson_con.save_as_json(path + '.json')
        return np.reshape(self.result_image, self.image_resolution), bson_con

    def get_result(self, cleanup: bool = True):
        if cleanup:
            self.cleanup_labelsets()
        return np.reshape(self.result_image, self.image_resolution), \
               bc.BsonContainer.fromValues(2,
                                           len(self.label_sets),
                                           len(self.segmentation_source),
                                           self.img_filename, {},
                                           self.label_sets,
                                           self.metadata)

    def cleanup_labelsets(self):
        # cleanup labelSets
        t = np.unique(self.result_image)
        t = list(t)
        if 0 not in t:
            t = [0] + t
        relabels = range(len(t) + 1)
        for a, b in zip(t, relabels):
            self.result_image[self.result_image == a] = b
        lookup_table = dict(zip([str(i) for i in t], relabels))
        # reconstruct the labelsets
        new_label_sets = {}
        for setname, labelset in self.label_sets.items():
            if setname in lookup_table.keys():
                new_label_sets[str(lookup_table[setname])] = list(labelset)
        for key, value in self.segmentation_source.items():
            self.segmentation_source[key] = list(value)
        self.label_sets = new_label_sets
