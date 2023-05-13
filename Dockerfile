FROM 3.10-bullseye
WORKDIR /node

RUN apt-get -y update

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

CMD ["python3","-u","scripts/main.py"]