import pytest

from labeling import LabelingData


def test():
    data = LabelingData.decode("./tests/test.lbl.json")
    assert data is not None
    assert len(data.labelSets) == data.numSets


if __name__ == "__main__":
    pytest.main()
