FROM dergunovdv/thanosar:latest

WORKDIR /app/
ADD requirements.txt .
RUN python3.7 -m pip install setuptools
RUN python3.7 -m pip install -r requirements.txt

ADD . .
EXPOSE 5000

CMD python3.7 wsgi.py
