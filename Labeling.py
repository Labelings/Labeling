import os
from typing import List, Callable, Tuple, T

import numpy as np
from PIL import Image
from tifffile import imread

import bsoncontainer as bc


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


class Labeling2:
    result_image = None
    image_resolution = ()
    label_value = 1
    label_sets = {"0": LabelSet()}
    list_of_unique_ids = []
    unique_map_id_id_label = {}
    segmentation_source = {}

    def __init__(self):
        self.result_image = None
        self.image_resolution = ()
        self.label_value = 1
        self.label_sets = {"0": LabelSet()}
        self.list_of_unique_ids = []
        self.unique_map_id_id_label = {}
        self.segmentation_source = {}

    @classmethod
    def from_file(cls, path: str):
        cls.labels = bc.BsonContainer.decode(path)
        cls.img = cls.labels.get_image()
        return cls

    @classmethod
    def from_file_withfunc(cls, path: str, func: Callable[[int], T]):
        cls.labels = bc.BsonContainer.decode_withfunc(path, func)
        cls.img = cls.labels.get_image()
        return cls

    @staticmethod
    def read_images(file_paths: list):
        return imread(file_paths)

    @classmethod
    def fromValues(cls, first_image: np.ndarray = np.zeros((512, 512), np.int8), source_id=None):
        transformer_list = {"0": 0}
        labeling = Labeling2()
        labeling.list_of_unique_ids = np.unique(first_image)
        labeling.image_resolution = first_image.shape
        # relabel first image to be the baseline
        with np.nditer(first_image, flags=["c_index"], op_flags=["readonly"]) as it:
            for uniqueLabel in labeling.list_of_unique_ids:
                if uniqueLabel != 0:
                    transformer_list[str(uniqueLabel)] = labeling.label_value
                    labeling.label_sets[str(labeling.label_value)] = LabelSet(source_id, labeling.label_value)
                    labeling.label_value += 1
        labeling.result_image = first_image.copy().flatten()

        with np.nditer(labeling.result_image, flags=["c_index"], op_flags=["readonly"]) as it:
            for val in it:
                labeling.result_image[it.index] = transformer_list[str(val)]

        unique_pairs = list(zip(np.repeat(0, len(labeling.list_of_unique_ids)), labeling.list_of_unique_ids))
        for x in range(len(labeling.list_of_unique_ids)):
            labeling.unique_map_id_id_label[unique_pairs[x]] = x
        labeling.list_of_unique_ids = labeling.list_of_unique_ids.tolist()
        return labeling

    def iterate_over_images(self, images: List[np.ndarray], source_ids=None):
        # iterate over all images
        for image, source_id in zip(images, source_ids):
            self.add_image(image, source_id)


    def add_image(self, image: np.ndarray, source_id=None):
        self.add_segments(image, (0, 0), source_id)

    def add_segments(self, patch: np.ndarray, position: Tuple, merge: dict = None, source_id=None):
        list_of_unique_ids = []
        segment_mapping = {}
        unique_map_id_id_label = {}
        temp = np.reshape(self.result_image, self.image_resolution)
        with np.nditer(patch, flags=["multi_index"], op_flags=["readonly"]) as it:
            for val in it:
                v = val.item()
                if v != 0:
                    pos = tuple(sum(x) for x in zip(it.multi_index, position))
                    if v not in list_of_unique_ids:
                        list_of_unique_ids.append(v)
                        self.label_sets[str(self.label_value)] = LabelSet(source_id, self.label_value)
                        segment_mapping[str(v)] = self.label_value
                    if (temp[pos], v) not in unique_map_id_id_label.keys():
                        unique_map_id_id_label[(temp[pos], v)] = self.label_value
                        self.label_sets[list(self.label_sets.keys())[-1]].add(self.label_value)
                        temp[pos] = self.label_value
                        self.label_value += 1
                    else:
                        label = unique_map_id_id_label[
                            (temp[pos], v)]
                        temp[pos] = label
        for id_id, label in unique_map_id_id_label.items():
            for key, value in self.label_sets.items():
                if id_id[0] in value.set:
                    self.label_sets[key].add(label)

        self.result_image = temp.flatten()
        return segment_mapping

    def add_segmentation_source(self, source_id, segment_id):
        if source_id is not None:
            if str(source_id) not in self.segmentation_source.keys():
                self.segmentation_source[str(source_id)] = set()
            self.segmentation_source[str(source_id)].add(segment_id)

    def save_result(self, path: str, cleanup:bool=False):
        if cleanup:
            self.cleanup_labelsets()
        img = Image.fromarray(np.reshape(self.result_image, self.image_resolution))
        img.save(path + '.tif', 'tiff')

        # convert labelSets to lists to save
        converted_labelsets = {}
        i = 0
        for x, y in self.label_sets.items():
            converted_labelsets[str(i)] = {"source": y.source, "set": list(y.set)}
            i += 1
        bson_con = bc.BsonContainer.fromValues(1, len(self.label_sets), os.path.splitext(path)[0] + '.tif', {},
                                               converted_labelsets, None)
        bson_con.encode_and_save(path + '.bson')
        # optional, just to easily content check
        bson_con.save_as_json(path + '.json')
        return np.reshape(self.result_image, self.image_resolution), bson_con

    def get_result(self, cleanup:bool=False):
        if cleanup:
            self.cleanup_labelsets()
        converted_labelsets = {}
        i = 0
        for x, y in self.label_sets.items():
            converted_labelsets[str(i)] = {"source": y.source, "set": list(y.set)}
            i += 1
        return np.reshape(self.result_image, self.image_resolution), \
               bc.BsonContainer.fromValues(1, len(self.label_sets), 'placeholder.tif', {}, converted_labelsets)

    def cleanup_labelsets(self):
        # cleanup labelSets
        t = np.unique(self.result_image)
        for setname, labelset in self.label_sets.items():
            labelset.set = set([x for x in labelset.set if x in t])
            self.label_sets[setname] = labelset
