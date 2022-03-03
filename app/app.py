import logging
import os
import timeit

import base64
import json


from PIL import Image
from io import BytesIO

from car_classifier import CarClassifier

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s')


def lambda_handler(event, context):
    image_bytes = event['body'].encode('utf-8')
    image = Image.open(BytesIO(base64.b64decode(image_bytes))).convert(mode='RGB')

    start = timeit.default_timer()
    classifier = CarClassifier()
    stop = timeit.default_timer()
    print('Model load time : {} msec'.format((stop - start) * 1000))

    start = timeit.default_timer()
    url = os.getenv("URL")
    if not url and 'url' in event:
        url = event['url']

    if url:
        infer_response = classifier.infer(url=url)
    else:
        infer_response = classifier.infer(image=image)
    stop = timeit.default_timer()
    print('Infer time : {} msec'.format((stop - start) * 1000))

    return {
        'statusCode': 200,
        'body': json.dumps(
            {
                "infer_response": infer_response,
            }
        )
    }
