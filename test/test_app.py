import unittest
import base64

from PIL.Image import Image

from app.app import lambda_handler


class TestApp(unittest.TestCase):
    def test_lambda_handler(self):
        with open("images/bmw-x4-1.jpeg", "rb") as img_file:
            image_binary = img_file.read()
            base64_encode = base64.b64encode(image_binary)
            byte_decode = base64_encode.decode('utf8')
        event = {"body": str(byte_decode)}
        # print(f'base64_encode={base64_encode}')
        print(f'event={event}')
        lambda_response = lambda_handler(event, None)
        print(f'lambda_response={lambda_response}')
        self.assertEqual(True, "BMW" in str(lambda_response))  # add assertion here


if __name__ == '__main__':
    unittest.main()
