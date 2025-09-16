FROM n8nio/n8n:latest

USER root

# Instalar Python + pip + bash + build tools necesarios
RUN apk add --no-cache python3 py3-pip bash build-base musl-dev python3-dev

# Crear un entorno virtual
RUN python3 -m venv /home/node/venv

# Copiamos requirements
COPY ../python/requirements.txt /tmp/requirements.txt

# Instalamos librerías en el venv
RUN /home/node/venv/bin/pip install --upgrade pip
RUN /home/node/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# Cambiar a usuario node
USER node

# Agregar venv al PATH
ENV PATH="/home/node/venv/bin:$PATH"

WORKDIR /home/node/.n8n
