import unittest
import numpy as np
from .utils import merge_arrs


class TestMerge(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMerge, self).__init__(*args, **kwargs)

    def test_merge_arrs(self):
        ar1_names = np.array(list(range(10)))
        ar2_names = ar1_names[::2]
        ar2_values = np.ones((len(ar2_names), ))
        ar2_mapped = merge_arrs(ar1_names, ar2_names, ar2_values)

        assert len(ar2_mapped) == len(ar1_names) - 1

if __name__ == "__main__":
    unittest.main()
