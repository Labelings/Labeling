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
    label_sets = {

    }

    def __init__(self):
        self.version = 1
        self.numSets = 0
        self.indexImg = ""
        self.labelMapping = {}
        self.label_sets = {}

    @classmethod
    def fromValues(cls, version: int = 1, numsets: int = 0, indeximg: str = "", labelmapping: dict = {},
                   label_sets: dict = {}):
        obj = BsonContainer()
        obj.numSets = numsets
        obj.indexImg = indeximg
        obj.labelMapping = labelmapping
        obj.label_sets = label_sets
        obj.version = version
        return obj

    @classmethod
    def fromDict(cls, data: dict):
        obj = BsonContainer()
        obj.version = data["version"]
        obj.numSets = data["numSets"]
        obj.indexImg = data["indexImg"]
        obj.labelMapping = data["labelMapping"]
        obj.label_sets = data["labelSets"]
        return obj

    def encode(self, path: str):
        data = bson.encode(vars(self))
        with open(path, 'wb+') as f:
            f.write(data)

    def encodewithfunc(self, path: str, func: Callable[[T], int]):
        for labelset in self.label_sets.values():
            i = 0
            for label in labelset:
                t = func(label)
                self.labelMapping[str(t)] = label
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
