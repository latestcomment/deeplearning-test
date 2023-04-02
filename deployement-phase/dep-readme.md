# How to run program

## Set Docker

### Build Flask Application image from docker file
go to deployement-phase directory

```
docker build -t exaheydemans/flask-model-deployement.v0 .
```

### Push image to docker hub
login to docker CLI

```
docker tag exaheydemans/flask-model-deployement.v0:latest latestcomment/pythorch-model:1.0
docker push latestcomment/pythorch-model:1.0
```

### Pull image to docker hub

```
docker pull postgres:15
docker pull latestcomment/pythorch-model:1.0
```

### Build docker container for Flask App and Postgres DB

```
docker-compose up -d
```

## Run test
go to directory test

run test.py
```
python3.10 test.py
```
for change image input

- open edit test.py

- edit line below to specific image

```
resp = requests.post("http://127.0.0.1:5000/predict", files={'file': open(<INPUT-IMAGE>, 'rb')})
```

for example
```
resp = requests.post("http://127.0.0.1:5000/predict", files={'file': open('automobile1.png', 'rb')})
```

## Check Postgres DB
go to postgres container

```
> docker exec -it deployement-phase_db_1 /bin/bash

> psql -U postgres

> SELECT * FROM model;
```
