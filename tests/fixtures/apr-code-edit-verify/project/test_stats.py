import unittest

from stats import mean, median


class TestStats(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(mean([2, 4, 6]), 4)

    def test_median(self):
        self.assertEqual(median([3, 1, 2]), 2)
        self.assertEqual(median([4, 1, 3, 2]), 2.5)


if __name__ == "__main__":
    unittest.main()
