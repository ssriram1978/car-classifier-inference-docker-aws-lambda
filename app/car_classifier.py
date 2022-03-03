import logging
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
import traceback
import sys
import timeit
from imageio import imread
import os
from pathlib import Path

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s')


class CarClassifier:
    def __init__(self):
        """
        Load Pytorch model.
            :return:
        """
        # Create the validation data transform object for augmenting the image and resizing the image to fit into an
        # existing pre-trained model.
        #
        self._transform = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                              ])

        """
        option - A: You could directly download your model from Model store.
        This adds additional run-time Latency in downloading the model.
        Example: self._imported_model = AutoModelForImageClassification.from_pretrained(
        "SriramSridhar78/sriram-car-classifier")
        """
        model_path = os.path.join(Path(__file__).parent.absolute(), "model")
        self._imported_model = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name_or_path=model_path)

    def infer(self, url="", image=None):
        infer_response = ""
        if (not url or not len(url)) and not image:
            return infer_response
        try:
            if len(url):
                img = imread(uri=url)
                img = Image.fromarray(img)
            else:
                img = image
            print("Tranforming the image...")
            img = self._transform(img)
            img = img.unsqueeze(0)
            print("Performing a forward pass on the model...")
            output = self._imported_model(img)
            softmax_val = output.logits.softmax(dim=1)

            # print(f'output.logits.softmax(dim=1) = {output.logits.softmax(dim=1)}')
            # print(f'torch.argmax(output.logits.softmax(dim=1)) = {torch.argmax(output.logits.softmax(dim=1))}')
            # print(f'torch.sum(output.logits.softmax(dim=1) = {torch.sum(output.logits.softmax(dim=1))}')
            # print(f'torch.max(output.logits.softmax(dim=1)) = {torch.max(output.logits.softmax(dim=1))}')
            # print(f'type(softmax_val) = {type(softmax_val)}')

            # print(f'torch.topk(softmax_val, k=3, dim=1) = {torch.topk(softmax_val, k=3, dim=1)}')
            scores, indices = torch.topk(softmax_val, k=2, dim=1)
            # Squeeze the dimensions. Example: scores = [[0.3, 0.5, 0.1]] 1,3, 1 => scores.squeeze() = [0.3, 0.5,
            # 0.1] 1, 3, 1
            scores = scores.squeeze()  # squeeze last dimension
            indices = indices.squeeze()  # squeeze last dimension
            infer_response_list = []
            for index, score in torch.stack((indices, scores), dim=1):
                infer_response_list.append('score={}, index={}.'.format(score, index))
                infer_response_list.append('Could be {} with probability score of {}.\n'.format(
                    self._imported_model.config.id2label[int(index)], score))
            infer_response = '\n'.join(infer_response_list)
        except:
            traceback.print_exception(*sys.exc_info())
        print("infer_response = {}".format(infer_response))
        return infer_response


if __name__ == "__main__":
    """
    Examples:
    url = 'https://i.cbc.ca/1.3232774.1442526016!/fileImage/httpImage/image.jpg_gen/derivatives/16x9_780/skinny-polar-bear-svalbard.jpg'
    url = 'https://www.drivespark.com/images/2021-03/bmw-x4-1.jpg'
    url = 'https://barrettjacksoncdn.azureedge.net/staging/carlist/items/Fullsize/Cars/200473/200473_Side_Profile_Web.jpg'
    url = 'https://target.scene7.com/is/image/Target/GUEST_6c637a55-eff0-49ad-a9ae-2057eb33d738?wid=488&hei=488&fmt=pjpeg'
    url = 'https://www.elephantnaturepark.org/wp-content/uploads/2020/04/94351084_2570825139826450_7563146999747313664_n-600x374.jpg'
    """
    url = os.getenv("URL")
    if not url:
        raise "Please define an environment variable with URL=<url_path>"
    start = timeit.default_timer()
    classifier = CarClassifier()
    stop = timeit.default_timer()
    logging.info('Model load time : {} msec'.format((stop - start) * 1000))

    start = timeit.default_timer()
    print(f'infer response = {classifier.infer(url=url)}')
    stop = timeit.default_timer()
    logging.info('Infer time : {} msec'.format((stop - start) * 1000))
