FROM python:3.10.6-buster

COPY requirements.txt /requirements.txt

RUN pip install -U pip
RUN pip install -r requirements.txt

COPY api /api
COPY ner_model /ner_model


CMD uvicorn api.api:app --host 0.0.0.0 --port $PORT
