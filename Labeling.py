from typing import List, Callable, T, Tuple

import numpy as np
from PIL import Image
from tifffile import imread

import bsoncontainer as bc


class Labeling:
    result_image = None
    image_resolution = ()
    label_value = 1
    label_sets = {"0": set()}
    list_of_unique_ids = []
    unique_map_id_id_label = {}

    def __init__(self):
        self.result_image = None
        self.image_resolution = ()
        self.label_value = 1
        self.label_sets = {"0": set()}
        self.list_of_unique_ids = []
        self.unique_map_id_id_label = {}

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
    def fromValues(cls, first_image: np.ndarray = np.zeros((512, 512))):
        transformer_list = {"0": 0}
        labeling = Labeling()
        labeling.list_of_unique_ids = np.unique(first_image)
        labeling.image_resolution = first_image.shape
        # relabel first image to be the baseline
        with np.nditer(first_image, flags=["c_index"], op_flags=["readonly"]) as it:
            for uniqueLabel in labeling.list_of_unique_ids:
                if uniqueLabel != 0:
                    transformer_list[str(uniqueLabel)] = labeling.label_value
                    labeling.label_sets[str(labeling.label_value)] = set()
                    labeling.label_sets[str(labeling.label_value)].add(labeling.label_value)
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

    def iterate_over_images(self, images: List[np.ndarray]):
        # iterate over all images
        for image in images:
            with np.nditer(image, flags=["c_index"], op_flags=["readonly"]) as it:
                for val in it:
                    if val.item() != 0:
                        # add new label set
                        if val.item() not in self.list_of_unique_ids:
                            self.list_of_unique_ids.append(val.item())
                            self.label_sets[str(self.label_value)] = set()
                            self.label_sets[str(self.label_value)].add(self.label_value)
                        # add a new pixel-value to all relevant sets
                        if (self.result_image[it.index], val.item()) not in self.unique_map_id_id_label.keys():
                            self.unique_map_id_id_label[(self.result_image[it.index], val.item())] = self.label_value
                            for key, value in self.label_sets.items():
                                if self.result_image[it.index] in value:
                                    self.label_sets[key].add(self.label_value)
                            self.label_sets[list(self.label_sets.keys())[-1]].add(self.label_value)
                            self.result_image[it.index] = self.label_value
                            self.label_value += 1
                        else:
                            label = self.unique_map_id_id_label[(self.result_image[it.index], val.item())]
                            self.result_image[it.index] = label

    def add_segments(self, patch: np.ndarray, position: Tuple, merge:dict=None):
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
                        self.label_sets[str(self.label_value)] = set()
                        self.label_sets[str(self.label_value)].add(self.label_value)
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
                if id_id[0] in value:
                    self.label_sets[key].add(label)

        self.result_image = temp.flatten()
        return segment_mapping

    def save_result(self, path: str):
        self.cleanup_labelsets()
        img = Image.fromarray(np.reshape(self.result_image, self.image_resolution))
        img.save(path + '.tif', 'tiff')

        # convert labelSets to lists to save
        convertedlabelsets = {}
        i = 0
        for x, y in self.label_sets.items():
            convertedlabelsets[str(i)] = list(y)
            i += 1

        bsonCon = bc.BsonContainer.fromValues(1, len(self.label_sets), 'placeholder.tif', {}, convertedlabelsets)
        bsonCon.encode_and_save(path + '.bson')
        # optional, just to easily content check
        bsonCon.save_as_json(path + '.json')
        return np.reshape(self.result_image, self.image_resolution), bsonCon

    def get_result(self):
        self.cleanup_labelsets()
        convertedlabelsets = {}
        i = 0
        for x, y in self.label_sets.items():
            convertedlabelsets[str(i)] = list(y)
            i += 1
        return np.reshape(self.result_image, self.image_resolution), \
               bc.BsonContainer.fromValues(1, len(self.label_sets), 'placeholder.tif', {}, convertedlabelsets)

    def cleanup_labelsets(self):
        # cleanup labelSets
        t = np.unique(self.result_image)
        for setname, labelset in self.label_sets.items():
            self.label_sets[setname] = set([x for x in labelset if x in t])
