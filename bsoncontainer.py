import json
from typing import Tuple, Callable, T

import bson
import numpy as np
from tifffile import imread


class BsonContainer:
    version = 1
    numSets = 0
    indexImg = ""
    labelMapping = {}
    labelSets = {

    }

    def __init__(self):
        self.version = 1
        self.numSets = 0
        self.indexImg = ""
        self.labelMapping = {}
        self.labelSets = {}

    @classmethod
    def fromValues(cls, version: int = 1, numsets: int = 0, indeximg: str = "", labelmapping: dict = {},
                   label_sets: dict = {}):
        obj = BsonContainer()
        obj.numSets = numsets
        obj.indexImg = indeximg
        obj.labelMapping = labelmapping
        obj.labelSets = label_sets
        obj.version = version
        return obj

    @classmethod
    def fromDict(cls, data: dict):
        obj = BsonContainer()
        obj.version = data["version"]
        obj.numSets = data["numSets"]
        obj.indexImg = data["indexImg"]
        obj.labelMapping = data["labelMapping"]
        obj.labelSets = data["labelSets"]
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
            return BsonContainer.fromValues(data["version"], data["numSets"], data["indexImg"], data["labelMapping"],
                                            data["labelSets"])

    @staticmethod
    def decode_withfunc(path, func: Callable[[int], T]):
        with open(path, 'rb') as f:
            data = bson.decode(f.read())
            transformed_labelsets = {}
            for key, labelset in data['labelSets'].items():
                transformed_labelsets[key] = set()
                for label in labelset:
                    transformed_labelsets[key].add(func(label))

            return BsonContainer.fromValues(data["version"], data["numSets"], data["indexImg"], data["labelMapping"],
                                            transformed_labelsets)

    def get_image(self):
        return imread(self.indexImg)


class Labeling:
    labels = BsonContainer.fromValues()
    img = None

    def __init__(self, dim: Tuple = None, x: int = None, y: int = None):
        if x and y is not None:
            self.img = np.zeros((x, y))
        if dim is not None:
            self.img = np.zeros(dim)
        else:
            self.img = np.zeros((1, 1))

    @classmethod
    def from_file(cls, path: str):
        cls.labels = BsonContainer.decode(path)
        cls.img = cls.labels.get_image()
        return cls

    @classmethod
    def from_file_withfunc(cls, path: str, func: Callable[[int], T]):
        cls.labels = BsonContainer.decode_withfunc(path, func)
        cls.img = cls.labels.get_image()
        return cls
