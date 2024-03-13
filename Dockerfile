FROM python:3.10.6-buster

COPY requirements.txt /requirements.txt

RUN pip install -U pip
RUN pip install -r requirements.txt

COPY ner_model /ner_model
COPY data /data
COPY api /api


CMD uvicorn api.api:app --host 0.0.0.0 --port $PORT
