import copy
import os
from typing import List, Callable, Tuple, T, Dict

import numpy as np
from PIL import Image
from tifffile import imread

from .LabelingData import LabelingData


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
        self.__image_resolution = shape
        self.__label_value = 0
        self.__pixel_value = 1
        self.label_sets = {"0": set()}
        self.__list_of_unique_ids = []
        self.__unique_map_id_id_label = {}
        self.__segmentation_source = {}
        self.metadata = None
        self.__img_filename = None

    @classmethod
    def from_file(cls, path: str):
        labeling = cls.__new__(cls)
        container = LabelingData.decode(path)
        labeling.result_image = container.get_image()
        labeling.__image_resolution = labeling.result_image.shape
        labeling.label_sets = container.labelSets
        labeling.metadata = container.metadata
        labeling.__img_filename = os.path.split(path)[1].replace(".lbl.json", ".tif")
        labeling.__segmentation_source = dict.fromkeys(range(container.numSources), range(container.numSources))
        return labeling

    # @classmethod
    # def from_file_withfunc(cls, path: str, func: Callable[[int], T]):
    #     cls.labels = bc.BsonContainer.decode_withfunc(path, func)
    #     cls.result_image = cls.labels.get_image()
    #     cls.img = cls.labels.get_image()
    #     return cls

    @staticmethod
    def read_images(file_paths: list) -> np.ndarray:
        return imread(file_paths)

    @classmethod
    def fromValues(cls, first_image: np.ndarray = np.zeros((512, 512), np.int8), source_id=None):
        labeling = Labeling()
        labeling.__list_of_unique_ids = np.unique(first_image)
        labeling.__image_resolution = first_image.shape
        labeling.result_image = np.zeros(first_image.shape, first_image.dtype)
        labeling.add_image(first_image, source_id)
        return labeling

    def iterate_over_images(self, images: List[np.ndarray], source_ids=None) -> List[Dict[str, LabelSet]]:
        # iterate over all images
        ret = []
        for image, source_id in zip(images, source_ids):
            ret.append(self.add_image(image, source_id))
        return ret

    def add_image(self, image: np.ndarray, source_id=None) -> Dict[str, LabelSet]:
        return self.add_segments(image, (0, 0), source_id=source_id)

    def add_segments(self, patch: np.ndarray, position: Tuple, merge: dict = None, source_id=None) -> Dict[
        str, LabelSet]:
        list_of_unique_labelvalues = set()
        segment_mapping = {}
        __unique_map_id_id_label = {}
        temp = np.reshape(self.result_image, self.__image_resolution)
        with np.nditer(patch, flags=["multi_index"], op_flags=["readonly"]) as it:
            for val in it:
                if val.item() != 0:
                    v = val.item()
                    pos = tuple(sum(x) for x in zip(it.multi_index, position))
                    if (temp[pos], v) not in __unique_map_id_id_label:

                        if v not in list_of_unique_labelvalues:
                            self.__label_value += 1
                            list_of_unique_labelvalues.add(v)
                        __unique_map_id_id_label[(temp[pos], v)] = (self.__pixel_value, self.__label_value)
                        if temp[pos] == 0:
                            self.label_sets[str(self.__pixel_value)] = set()
                            self.label_sets[str(self.__pixel_value)].add(self.__label_value)
                            self.add_segmentation_source(source_id, self.__label_value)
                            segment_mapping[str(self.__pixel_value)] = LabelSet(source_id, self.__label_value)
                        else:
                            labelset = copy.deepcopy(self.label_sets[str(temp[pos])])
                            labelset.add(self.__label_value)
                            if str(self.__pixel_value) not in self.label_sets:
                                self.label_sets[str(self.__pixel_value)] = labelset
                            self.add_segmentation_source(source_id, self.__label_value)

                        temp[pos] = self.__pixel_value
                        self.__pixel_value += 1

                    else:
                        temp[pos] = __unique_map_id_id_label[(temp[pos], v)][0]

        self.result_image = temp.flatten()
        return segment_mapping

    def add_metadata(self, data) -> None:
        self.metadata = data

    def add_segmentation_source(self, source_id, __label_value) -> None:
        if source_id is not None:
            if str(source_id) not in self.__segmentation_source.keys():
                self.__segmentation_source[str(source_id)] = set()
            self.__segmentation_source[str(source_id)].add(__label_value)

    def save_result(self, path: str, cleanup: bool = False):
        if cleanup:
            self.__cleanup_labelsets()
        img = Image.fromarray(np.reshape(self.result_image, self.__image_resolution))
        path, filename = os.path.split(path)
        img.save(os.path.join(path, filename + '.tif'), 'tiff')
        self.__img_filename = filename + '.tif'
        label_data = LabelingData.fromValues(2, len(self.label_sets), len(self.__segmentation_source),
                                             self.__img_filename, {},
                                             {key: list(value) for (key, value) in
                                              self.label_sets.items()}, self.metadata)
        label_data.save_as_json(os.path.join(path, filename + '.lbl.json'))
        return np.reshape(self.result_image, self.__image_resolution), label_data

    def get_result(self, cleanup: bool = False) -> (np.array, LabelingData):
        if cleanup:
            self.__cleanup_labelsets()
        return np.reshape(self.result_image, self.__image_resolution), \
               LabelingData.fromValues(2,
                                       len(self.label_sets),
                                       len(self.__segmentation_source),
                                       self.__img_filename, {},
                                       {key: list(value) for (key, value) in self.label_sets.items()},
                                       self.metadata)

    def __cleanup_labelsets(self) -> None:
        # cleanup labelSets
        _, idx = np.unique(self.result_image, return_index=True)
        t = list(self.result_image[np.sort(idx)])
        if 0 not in t:
            t.insert(0, 0)
        relabels = range(len(t) + 1)

        temp = np.zeros(self.result_image.shape, np.int8)
        for a, b in zip(t, relabels):
            temp[self.result_image == a] = b
        self.result_image = temp
        lookup_table = dict(zip([str(i) for i in t], relabels))
        # reconstruct the labelsets
        new_label_sets = {}
        segments = self.__segment_fragment_mapping().keys()
        segment_remapping = dict(zip(segments, range(1, len(segments) + 2)))
        for setname, labelset in self.label_sets.items():
            if setname in lookup_table.keys():
                new_label_sets[str(lookup_table[setname])] = [segment_remapping[x] for x in list(labelset)]
        for key, value in self.__segmentation_source.items():
            self.__segmentation_source[key] = list(value)
        self.label_sets = new_label_sets

    def __segment_fragment_mapping(self) -> dict:
        segment_to_fragment = {}
        for key, value in self.label_sets.items():
            for v in value:
                if v not in segment_to_fragment:
                    segment_to_fragment[v] = set()
                segment_to_fragment[v].add(int(key))
        return segment_to_fragment

    def remove_segment(self, segment_number: int) -> None:
        segment_to_fragment = self.__segment_fragment_mapping()
        fragments = segment_to_fragment.pop(segment_number)
        transformation_list = []
        for fragment in fragments:
            self.label_sets[str(fragment)].remove(segment_number)
        for fragment in fragments:
            for fragment_id, segment_list in self.label_sets.items():
                if set(self.label_sets[str(fragment)]) == set(segment_list):
                    # not (A,B),(B,A) already in list and not (A,A)
                    if not any(elem in transformation_list for elem in
                               [(int(fragment_id), fragment), (fragment, int(fragment_id))]) and fragment != int(
                        fragment_id):
                        transformation_list.append((fragment, int(fragment_id)))
        for transformer in transformation_list:
            np.place(self.result_image, self.result_image == transformer[0], transformer[1])

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False
