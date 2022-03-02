import logging
import os
import timeit

# import torch
# import torchvision
import base64
import json
# import numpy as np

# import torch.nn as nn
# import torch.nn.functional as F

from PIL import Image
from io import BytesIO

from app.car_classifier import CarClassifier

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s')

"""
image_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(320, 100)
        self.bn1 = nn.BatchNorm1d(100)

        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)

        self.smax = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 320)
        x = self.bn1(F.relu(self.fc1(x)))
        x = F.dropout(x, training=self.training)

        x = self.bn2(F.relu(self.fc2(x)))
        x = F.dropout(x, training=self.training)

        return F.softmax(self.smax(x), dim=-1)


model_file = '/opt/ml/model'
model = Net()
model.load_state_dict(torch.load(model_file))
model.eval()
"""


def lambda_handler(event, context):
    image_bytes = event['body'].encode('utf-8')
    image = Image.open(BytesIO(base64.b64decode(image_bytes))).convert(mode='RGB')
    # image = image.resize((28, 28))

    # Original pytorch example.
    # probabilities = model.forward(image_transforms(np.array(image)).reshape(-1, 1, 28, 28))
    # label = torch.argmax(probabilities).item()

    """
        Examples:
        url = 'https://i.cbc.ca/1.3232774.1442526016!/fileImage/httpImage/image.jpg_gen/derivatives/16x9_780/skinny-polar-bear-svalbard.jpg'
        url = 'https://www.drivespark.com/images/2021-03/bmw-x4-1.jpg'
        url = 'https://barrettjacksoncdn.azureedge.net/staging/carlist/items/Fullsize/Cars/200473/200473_Side_Profile_Web.jpg'
        url = 'https://target.scene7.com/is/image/Target/GUEST_6c637a55-eff0-49ad-a9ae-2057eb33d738?wid=488&hei=488&fmt=pjpeg'
        url = 'https://www.elephantnaturepark.org/wp-content/uploads/2020/04/94351084_2570825139826450_7563146999747313664_n-600x374.jpg'
        """
    start = timeit.default_timer()
    classifier = CarClassifier()
    stop = timeit.default_timer()
    logging.info('Model load time : {} msec'.format((stop - start) * 1000))

    start = timeit.default_timer()
    url = os.getenv("URL")
    if url:
        infer_response = classifier.infer(url=url)
    else:
        infer_response = classifier.infer(image=image)
    stop = timeit.default_timer()
    logging.info('Infer time : {} msec'.format((stop - start) * 1000))

    return {
        'statusCode': 200,
        'body': json.dumps(
            {
                "infer_response": infer_response,
            }
        )
    }
