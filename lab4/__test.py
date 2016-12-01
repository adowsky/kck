from lab4 import process_image
import unittest


class TestStringMethods(unittest.TestCase):
    tests = [
        {"filename": "bemole.jpg", "result": 26},
        {"filename": "odetojoy.jpg", "result": 19},
        {"filename": "jingle.jpg", "result": 12},
    ]

    def test_instances(self):
        for test in self.tests:
            self.assertEqual(process_image(False, test["filename"]), test["result"])

if __name__ == '__main__':
    unittest.main()