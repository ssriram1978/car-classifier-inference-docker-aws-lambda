import os
import timeit
import unittest
# Imports PIL module
from PIL import Image
from app.car_classifier import CarClassifier


class TestCarClassifier(unittest.TestCase):

    def infer(self, url="", image=None):
        start = timeit.default_timer()
        classifier = CarClassifier()
        stop = timeit.default_timer()
        print('Model load time : {} msec'.format((stop - start) * 1000))

        start = timeit.default_timer()
        if url and len(url):
            infer_response = classifier.infer(url=url)
        else:
            infer_response = classifier.infer(image=image)
        print(f'infer_response={infer_response}')
        stop = timeit.default_timer()
        print('Infer time : {} msec'.format((stop - start) * 1000))
        return infer_response

    def test_url(self):
        url = 'https://www.drivespark.com/images/2021-03/bmw-x4-1.jpg'
        self.assertEqual(True, 'BMW' in self.infer(url))
        url = 'https://barrettjacksoncdn.azureedge.net/staging/carlist/items/Fullsize/Cars/200473' \
              '/200473_Side_Profile_Web.jpg '
        self.assertEqual(True, 'Ferrari' in self.infer(url))
        url = 'https://target.scene7.com/is/image/Target/GUEST_6c637a55-eff0-49ad-a9ae-2057eb33d738?wid=488&hei=488' \
              '&fmt=pjpeg '
        self.assertEqual(True, 'Hummer' in self.infer(url))

    def test_image(self):
        # open method used to open different extension image file
        image = Image.open("images/bmw-x4-1.jpeg")
        self.assertEqual(True, 'BMW' in self.infer(image=image))
        image = Image.open("images/ferrari.jpeg")
        self.assertEqual(True, 'Ferrari' in self.infer(image=image))
        image = Image.open("images/hummer_toy.jpeg")
        self.assertEqual(True, 'Hummer' in self.infer(image=image))


if __name__ == '__main__':
    unittest.main()
