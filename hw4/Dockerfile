FROM svizor/zoomcamp-model:mlops-3.10.0-slim

WORKDIR /app

RUN pip install -U pip
RUN pip install pipenv


COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "starter.py", "constants.py", "utils.py", "./" ]

ENTRYPOINT [ "python",  "starter.py"]
