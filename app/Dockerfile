FROM public.ecr.aws/lambda/python:3.9

#COPY app.py requirements.txt ./
#COPY model /opt/ml/model

ADD . ./
WORKDIR ./

RUN python3.9 -m pip install -r requirements.txt -t .

CMD ["app.lambda_handler"]
