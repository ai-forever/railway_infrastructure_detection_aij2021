FROM cr.msk.sbercloud.ru/aicloud-base-images/horovod-cuda10.1-tf2.3.0

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py ./
ENV PYTHONPATH detection/yolov5/
CMD ["python3", "/app/solution.py"]