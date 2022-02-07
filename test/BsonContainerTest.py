import unittest

from labeling.Labeling import Labeling


class MyTestCase(unittest.TestCase):
    def test_equality(self):
        label1 = Labeling.from_file("example2.bson")
        img1, cont1 = label1.get_result()
        label2 = Labeling.from_file("example2.bson")
        img2, cont2 = label2.get_result()
        self.assertEqual(vars(cont1), vars(cont2))
        self.assertTrue(cont1 == cont2)
        cont2.numSources = 42
        self.assertNotEqual(vars(cont1), vars(cont2))
        self.assertTrue(cont1 == cont2)
        cont2.numSets = 42
        self.assertNotEqual(vars(cont1), vars(cont2))
        self.assertFalse(cont1 == cont2)


if __name__ == '__main__':
    unittest.main()
