FROM python:3.6-buster

RUN mkdir /autobazaar && \
    mkdir /input && \
    mkdir /output && \
    ln -s /input /autobazaar/input && \
    ln -s /output /autobazaar/output

# Copy code
COPY setup.py README.md HISTORY.md MANIFEST.in /autobazaar/

# Install project
RUN pip3 install -e /autobazaar && pip install ipdb

COPY autobazaar /autobazaar/autobazaar

CMD ["echo", "Usage: docker run -ti -u$UID -v /path/to/input:/input -v /path/to/output:/output mlbazaar/autobazaar abz OPTIONS"]
