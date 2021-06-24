import unittest
import BotometerRequests


class MyTestCase(unittest.TestCase):
    def test_replace_missed(self):
        br = BotometerRequests.BotometerRequests()
        old = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        new = [1, 3, 4, 5, 7, 10]
        presumed = [9, 9, 9, 9, 9, 9]
        null = -1

        res = br.replace_missed(old, new, presumed, null)
        self.assertListEqual([9, -1, 9, 9, 9, -1, 9, -1, -1, 9], res)

    def test_find_missed_and_insert_at_indices(self):
        br = BotometerRequests.BotometerRequests()
        old = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        new = [1, 3, 4, 5, 7, 10]
        res = br.find_missed(old, new)
        res2 = br.insert_at_indices(new, res, -1)

        self.assertListEqual([1, -1, 3, 4, 5, -1, 7, -1, -1, 10], res2)

    def test_segment_list(self):
        br = BotometerRequests.BotometerRequests()
        to_seg = [x for x in range(1, 11)]
        res = br.segment_list(to_seg, 2)
        expected = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]

        self.assertListEqual(expected, res)

    def test_flatten(self):
        br = BotometerRequests.BotometerRequests()
        to_flatten = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        res = br.flatten(to_flatten)
        expected = [x for x in range(1, 11)]

        self.assertListEqual(expected, res)



if __name__ == '__main__':
    unittest.main()
