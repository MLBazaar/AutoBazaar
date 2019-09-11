FROM python:3.6-buster

RUN mkdir /autobazaar && \
    mkdir /BTB && \
    mkdir /MLBlocks && \
    mkdir /abz && \
    ln -s /input /abz/input && \
    ln -s /output /abz/output

# Copy code
COPY setup.py README.md HISTORY.md MANIFEST.in /autobazaar/
COPY BTB/setup.py BTB/README.md BTB/HISTORY.md BTB/MANIFEST.in /BTB/
COPY MLBlocks/setup.py MLBlocks/README.md MLBlocks/HISTORY.md MLBlocks/MANIFEST.in /MLBlocks/

# Install project
RUN pip3 install -e /autobazaar && \
    pip3 install -e /BTB && \
    pip3 install -e /MLBlocks && \
    pip3 install ipdb

COPY autobazaar /autobazaar/autobazaar
COPY BTB/btb /BTB/btb
COPY MLBlocks/mlblocks /MLBlocks/mlblocks

WORKDIR /abz

CMD ["echo", "Usage: docker run -ti -u$UID -v $(pwd):/abz mlbazaar/autobazaar abz OPTIONS"]
