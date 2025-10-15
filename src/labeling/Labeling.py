import copy
import os
from typing import List, Tuple, Dict, Any

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
        labeling.__segmentation_source = dict.fromkeys(
            range(container.numSources), range(container.numSources)
        )
        return labeling

    @staticmethod
    def read_images(file_paths: list) -> np.ndarray:
        return imread(file_paths)

    @classmethod
    def fromValues(
        cls, first_image: np.ndarray = np.zeros((512, 512), np.int8), source_id=None
    ):
        labeling = Labeling()
        labeling.__list_of_unique_ids = np.unique(first_image)
        labeling.__image_resolution = first_image.shape
        labeling.result_image = np.zeros(first_image.shape, first_image.dtype)
        labeling.add_image(first_image, source_id)
        return labeling

    def iterate_over_images(
        self, images: List[np.ndarray], source_ids=None
    ) -> List[Dict[str, LabelSet]]:
        """
        Convenience function to add multiple original segmentations to an image.
        Segmentations need to be of the same size.
        :param images:
        :param source_ids:
        :return:
        """
        # iterate over all images
        ret = []
        for image, source_id in zip(images, source_ids):
            ret.append(self.add_image(image, source_id))
        return ret

    def add_image(self, image: np.ndarray, source_id=None) -> Dict[str, LabelSet]:
        """
        Add a single segmentation image to the Labeling.
        :param image:
        :param source_id:
        :return:
        """
        return self.add_segments(image, (0, 0), source_id=source_id)

    def add_segments(
        self, patch: np.ndarray, position: Tuple, merge: dict = None, source_id=None
    ) -> Dict[str, LabelSet]:
        """
        The main function of the library.
        Integrates a patch at the specified position.
        It iterates over each pixel of the patch and updates
        the Labeling accordingly, adding a new segment and creating fragments
        on demand.
        :param patch: a nD labeling patch. a patch can also be a full image in this case
        :param position: the nD position the patch is being inserted in. the
        position is "upperleft corner" i.e. it corresponds to (0,0) in the patch
        :param merge: unused. meant to hold additional information that refer to each
        patch pixels value
        :param source_id: a reference value for all segments that are added for future
        tracking
        :return: the mapping of fragment id to segment id
        """
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
                        __unique_map_id_id_label[(temp[pos], v)] = (
                            self.__pixel_value,
                            self.__label_value,
                        )
                        if temp[pos] == 0:
                            self.label_sets[str(self.__pixel_value)] = set()
                            self.label_sets[str(self.__pixel_value)].add(
                                self.__label_value
                            )
                            self.add_segmentation_source(source_id, self.__label_value)
                            segment_mapping[str(self.__pixel_value)] = LabelSet(
                                source_id, self.__label_value
                            )
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
        """
        Optional
        Add any additional metadata to the Labeling that might be necessary.
        :param data:
        :return: None
        """
        self.metadata = data

    def add_segmentation_source(self, source_id: str, label_value: Any) -> None:
        """
        Optional
        Adds a mapping of user-defined source_id to a value(should be either segment or fragment id)
        :param source_id:
        :param label_value:
        :return: None
        """
        if source_id is not None:
            if str(source_id) not in self.__segmentation_source.keys():
                self.__segmentation_source[str(source_id)] = set()
            self.__segmentation_source[str(source_id)].add(label_value)

    def save_result(self, path: str, cleanup: bool = False):
        """

        :param path: the path to save the Labeling to
        :param cleanup: if the metadata should be cleaned up of all no longer valid data
        :return: the merged image, a LabelingData object
        """
        if cleanup:
            self.__cleanup_labelsets()
        img = Image.fromarray(np.reshape(self.result_image, self.__image_resolution))
        path, filename = os.path.split(path)
        img.save(os.path.join(path, filename + ".tif"), "tiff")
        self.__img_filename = filename + ".tif"
        label_data = LabelingData.fromValues(
            2,
            len(self.label_sets),
            len(self.__segmentation_source),
            self.__img_filename,
            {},
            {key: list(value) for (key, value) in self.label_sets.items()},
            self.metadata,
        )
        label_data.save_as_json(os.path.join(path, filename + ".lbl.json"))
        return np.reshape(self.result_image, self.__image_resolution), label_data

    def get_result(self, cleanup: bool = False) -> (np.array, LabelingData):
        """
        This returns the current state of the Labeling image.
        and it's corresponding metadata.
        :param cleanup:
        :return: the merged image, a LabelingData object
        """
        if cleanup:
            self.__cleanup_labelsets()
        return np.reshape(
            self.result_image, self.__image_resolution
        ), LabelingData.fromValues(
            2,
            len(self.label_sets),
            len(self.__segmentation_source),
            self.__img_filename,
            {},
            {key: list(value) for (key, value) in self.label_sets.items()},
            self.metadata,
        )

    def __cleanup_labelsets(self) -> None:
        """
        Removes all fragments that are in the dict but no longer in
        the containing image. It also removes all segments
        that have no associated fragments from the list.
        :return:
        """
        values, indices = np.unique(self.result_image, return_index=True)
        # We want list of unique values sorted by their appearance in result_image.
        # We can do this by zipping index and value, then passing to sorted.
        # sorted() on a tuple sorts element-by-element, meaning the index will
        # be the deciding factor in the sort.
        t = [value for index, value in sorted(zip(indices, values))]
        if 0 not in t:
            t.insert(0, 0)
        relabels = range(len(t) + 1)

        temp = np.zeros(self.result_image.shape, self.result_image.dtype)
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
                new_label_sets[str(lookup_table[setname])] = set(
                    [segment_remapping[x] for x in list(labelset)]
                )
        for key, value in self.__segmentation_source.items():
            self.__segmentation_source[key] = list(value)
        self.label_sets = new_label_sets

    def __segment_fragment_mapping(self) -> dict:
        """
        Returns the current representation of a segment-fragments mapping.
        This is not a copy of the values.
        :return: a dict[int,set(int)]
        """
        segment_to_fragment = {}
        for key, value in self.label_sets.items():
            for v in value:
                if v not in segment_to_fragment:
                    segment_to_fragment[v] = set()
                segment_to_fragment[v].add(int(key))
        return segment_to_fragment

    def remove_segment(self, segment_number: int) -> None:
        """
        Removes a single segment from the Labeling object,
        rearranging all involved fragments.
        :param segment_number: the segment to remove from the Labeling.
        :return:
        """
        segment_to_fragment = self.__segment_fragment_mapping()
        fragments = segment_to_fragment.pop(segment_number)
        transformation_list = []
        for fragment in fragments:
            self.label_sets[str(fragment)].remove(segment_number)
        for fragment in fragments:
            for fragment_id, segment_list in self.label_sets.items():
                if set(self.label_sets[str(fragment)]) == set(segment_list):
                    # not (A,B),(B,A) already in list and not (A,A)
                    if not any(
                        elem in transformation_list
                        for elem in [
                            (int(fragment_id), fragment),
                            (fragment, int(fragment_id)),
                        ]
                    ) and fragment != int(fragment_id):
                        transformation_list.append((fragment, int(fragment_id)))
        for transformer in transformation_list:
            np.place(
                self.result_image, self.result_image == transformer[0], transformer[1]
            )

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False
