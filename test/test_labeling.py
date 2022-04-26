import pytest

import numpy as np
from scipy.ndimage import rotate
from labeling import Labeling as lb


def test_add_multiple_images():
    image = np.ones((3, 3))
    labeling = lb.fromValues(np.zeros((3, 3), np.int32))
    for i in range(5):
        labeling.add_image(image, str(i))

    img, labeling = labeling.get_result(True)

    assert all(elem in list(labeling.labelSets.keys()) for elem in [str(i) for i in list(np.unique(img))])
    assert len(set([item for sublist in labeling.labelSets.values() for item in sublist])) == 5


def test_add_multiple_images_diff_values():
    image = np.ones((3, 3))
    labeling = lb.fromValues(np.zeros((3, 3), np.int32))
    for i in range(1, 5):
        image[0] = i * 3
        image[1] = i * 3 + 1
        image[2] = i * 3 + 2
        labeling.add_image(image, str(i))

    img, labeling = labeling.get_result(True)
    assert all(elem in list(labeling.labelSets.keys()) for elem in [str(i) for i in list(np.unique(img))])
    assert len(set([item for sublist in labeling.labelSets.values() for item in sublist])) == 12


def test_load_from_file():
    labeling = lb.from_file("./test/test.lbl.json")
    assert labeling.get_result()[0] is not None
    assert labeling.get_result()[1] is not None

def test_save_to_file():
    labeling = lb.from_file("./test/test.lbl.json")
    assert labeling.get_result()[0] is not None
    assert labeling.get_result()[1] is not None
    labeling.save_result("./test/test", True)


def test_add_delete():
    example2_images = [np.array([[0, 1, 1, 0],
                                 [0, 1, 1, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 0]])]
    example2_images.append(
        rotate(example2_images[0].copy(), angle=90, reshape=False, mode="constant", cval=0))
    example2_images.append(
        rotate(example2_images[1].copy(), angle=90, reshape=False, mode="constant", cval=0))

    merger = lb.fromValues(np.zeros((4, 4), np.int32))
    merger.iterate_over_images(example2_images, [str(i) for i in list(range(1, len(example2_images) + 1))])
    merger.remove_segment(1)
    img, labeling = merger.get_result(True)

    merger2 = lb.fromValues(np.zeros((4, 4), np.int32))
    merger2.iterate_over_images(example2_images[1:], [str(i) for i in list(range(1, len(example2_images) + 1))])

    img2, labeling2 = merger2.get_result(True)
    assert np.all(np.equal(img, img2))
    assert labeling == labeling2

    merger = lb.fromValues(np.zeros((4, 4), np.int32))
    merger.iterate_over_images(example2_images, [str(i) for i in list(range(1, len(example2_images) + 1))])
    merger.remove_segment(2)
    img, labeling = merger.get_result(True)

    merger2 = lb.fromValues(np.zeros((4, 4), np.int32))
    merger2.add_image(example2_images[0], 1)
    merger2.add_image(example2_images[2], 2)
    img2, labeling2 = merger2.get_result(True)
    assert np.all(np.equal(img, img2))
    assert labeling == labeling2


if __name__ == '__main__':
    pytest.main(["--cov=labeling test/"])
