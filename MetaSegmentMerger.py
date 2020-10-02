import numpy as np
from PIL import Image
from tifffile import imread
from typing import List

import bsoncontainer as bc


class MetaSegmentMerger:
    patch_size = None
    result_image = None
    image_resolution = ()
    label_value = 1
    label_sets = {"0": set()}
    list_of_unique_ids = []
    unique_map_id_id_label = {}

    def __init__(self):
        self.patch_size = None
        self.result_image = None
        self.image_resolution = ()
        self.label_value = 1
        self.label_sets = {"0": set()}
        self.list_of_unique_ids = []
        self.unique_map_id_id_label = {}

    @staticmethod
    def read_images(file_paths: list):
        return imread(file_paths)

    @classmethod
    def fromValues(cls, patch_size: int = 64, first_image: np.ndarray = np.zeros((512, 512))):
        transformer_list = {"0": 0}
        obj = MetaSegmentMerger()
        obj.patch_size = patch_size
        obj.list_of_unique_ids = np.unique(first_image)
        obj.image_resolution = first_image.shape
        # relabel first image to be the baseline
        with np.nditer(first_image, flags=["c_index"], op_flags=["readonly"]) as it:
            for uniqueLabel in obj.list_of_unique_ids:
                if uniqueLabel != 0:
                    transformer_list[str(uniqueLabel)] = obj.label_value
                    obj.label_sets[str(obj.label_value)] = set()
                    obj.label_sets[str(obj.label_value)].add(obj.label_value)
                    obj.label_value += 1
        obj.result_image = first_image.copy().flatten()

        with np.nditer(obj.result_image, flags=["c_index"], op_flags=["readonly"]) as it:
            for val in it:
                obj.result_image[it.index] = transformer_list[str(val)]

        unique_pairs = list(zip(np.repeat(0, len(obj.list_of_unique_ids)), obj.list_of_unique_ids))
        for x in range(len(obj.list_of_unique_ids)):
            obj.unique_map_id_id_label[unique_pairs[x]] = x
        obj.list_of_unique_ids = obj.list_of_unique_ids.tolist()

        return obj

    def cleanup_labelsets(self):
        # cleanup labelSets
        t = np.unique(self.result_image)
        for setname, labelset in self.label_sets.items():
            self.label_sets[setname] = set([x for x in labelset if x in t])

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
                            self.result_image[it.index] = self.label_value
                            self.label_value += 1
                        else:
                            label = self.unique_map_id_id_label[(self.result_image[it.index], val.item())]
                            self.result_image[it.index] = label

    def add_segments(self, patch: np.ndarray, x: int, y: int):
        temp = np.reshape(self.result_image, self.image_resolution)
        with np.nditer(patch, flags=["multi_index"], op_flags=["readonly"]) as it:
            for val in it:
                if val.item() != 0:
                    if val.item() not in self.list_of_unique_ids:
                        self.list_of_unique_ids.append(val.item())
                        self.label_sets[str(self.label_value)] = set()
                        self.label_sets[str(self.label_value)].add(self.label_value)
                    if (temp[x + it.multi_index[0], y + it.multi_index[1]], val.item()) not in self.unique_map_id_id_label.keys():
                        self.unique_map_id_id_label[
                            (temp[x + it.multi_index[0], y + it.multi_index[1]], val.item())] = self.label_value
                        for key, value in self.label_sets.items():
                            if temp[x + it.multi_index[0], y + it.multi_index[1]] in value:
                                self.label_sets[key].add(self.label_value)
                        temp[x + it.multi_index[0], y + it.multi_index[1]] = self.label_value
                        self.label_value += 1
                    else:
                        label = self.unique_map_id_id_label[(temp[x + it.multi_index[0], y + it.multi_index[1]], val.item())]
                        temp[x + it.multi_index[0], y + it.multi_index[1]] = label
        self.result_image = temp.flatten()

    def save_result(self, path: str):
        self.cleanup_labelsets()
        img = Image.fromarray(np.reshape(self.result_image, (self.image_resolution[0], self.image_resolution[1])))
        img.save(path + '.tif', 'tiff')

        # convert labelSets to lists to save
        convertedlabelSets = {}
        i = 0
        for x, y in self.label_sets.items():
            convertedlabelSets[str(i)] = list(y)
            i += 1

        labeling = bc.Labeling((self.image_resolution[0], self.image_resolution[1]))
        labeling.img = img
        labeling.labels.label_sets = convertedlabelSets
        labeling.labels.numSets = len(self.label_sets)
        labeling.labels.indexImg = path + '.tif'
        labeling.labels.encode(path + '.bson')
        #optional, just to easily content check
        labeling.labels.save_as_json(path + '.json')
