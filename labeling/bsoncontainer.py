import json
import os
from typing import Callable, T

import bson
from tifffile import imread


class BsonContainer(dict):

    def __init__(self):
        self.version = 2
        self.numSets = 0
        self.numSources = 0
        self.indexImg = ""
        self.labelMapping = {}
        self.labelSets = {}
        self.metadata = {}

    @classmethod
    def fromValues(cls, version: int = 2, num_sets: int = 0, num_sources: int = 0, index_img: str = "",
                   label_mapping: dict = {},
                   label_sets: dict = {}, metadata: dict() = None):
        obj = BsonContainer()
        obj.numSets = num_sets
        obj.numSources = num_sources
        obj.indexImg = index_img
        obj.labelMapping = label_mapping
        obj.labelSets = label_sets
        obj.version = version
        if metadata is not None:
            obj.metadata = metadata
        return obj

    @classmethod
    def fromDict(cls, data: dict):
        obj = BsonContainer()
        obj.version = data["version"]
        obj.numSets = data["numSets"]
        obj.numSources = data["numSources"]
        obj.indexImg = data["indexImg"]
        obj.labelMapping = data["labelMapping"]
        obj.labelSets = data["labelSets"]
        obj.metadata = data["metadata"]
        return obj

    def encode(self):
        return bson.encode(vars(self))

    def encode_and_save(self, path: str):
        data = bson.encode(vars(self))
        with open(path, 'wb+') as f:
            f.write(data)

    def encodewithfunc(self, path: str, func: Callable[[T], int]):
        for labelset in self.labelSets.values():
            i = 0
            for label in labelset:
                t = func(label)
                labelset[i] = t
                i += 1

        data = bson.encode(vars(self))
        with open(path, 'wb+') as f:
            f.write(data)

    def save_as_json(self, path: str):
        with open(path, 'w') as outfile:
            json.dump(vars(self), outfile)

    @staticmethod
    def decode(path: str):
        with open(path, 'rb') as f:
            data = bson.decode(f.read())
            data["indexImg"] = os.path.join(os.path.dirname(os.path.realpath(f.name)), data["indexImg"])
            if "metadata" not in data.keys():
                return BsonContainer.fromValues(data["version"], data["numSets"], data["numSources"], data["indexImg"],
                                                data["labelMapping"],
                                                data["labelSets"])
            return BsonContainer.fromValues(data["version"], data["numSets"], data["numSources"], data["indexImg"],
                                            data["labelMapping"], data["labelSets"], data["metadata"])

    @staticmethod
    def decode_withfunc(path, func: Callable[[int], T]):
        with open(path, 'rb') as f:
            data = bson.decode(f.read())
            transformed_labelsets = {}
            for key, labelset in data['labelSets'].items():
                transformed_labelsets[key] = set()
                for label in labelset:
                    transformed_labelsets[key].add(func(label))

            if "metadata" not in data.keys():
                return BsonContainer.fromValues(data["version"], data["numSets"], data["numSources"], data["indexImg"],
                                                data["labelMapping"], data["labelSets"])
            return BsonContainer.fromValues(data["version"], data["numSets"], data["numSources"], data["indexImg"],
                                            data["labelMapping"], data["labelSets"], data["metadata"])

    def get_image(self):
        return imread(self.indexImg)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.labelSets == other.labelSets and self.numSets == other.numSets and \
                   self.labelMapping == other.labelMapping
        else:
            return False
