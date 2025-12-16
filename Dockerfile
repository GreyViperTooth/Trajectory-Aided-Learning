FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev python3-tk python3.10-tk tk \
    xvfb x11vnc fluxbox novnc websockify \
    libgl1-mesa-glx libglu1-mesa mesa-utils python3-opengl \
    git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/TrajectoryAidedLearning
COPY . /app/TrajectoryAidedLearning

RUN python3 -m pip install --no-cache-dir --upgrade pip==23.2.1 setuptools==59.5.0 wheel==0.41.2 && \
    python3 -m pip install --no-cache-dir \
        torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    python3 -m pip install --no-cache-dir \
        gym==0.26.2 pyglet==1.5.27 pillow==9.5.0 pyyaml==6.0 numpy==1.24.4 \
        matplotlib==3.8.2 numba==0.58.1 scipy==1.10.1 cma==4.4.1 && \
    python3 -m pip install --no-cache-dir -e .

ENV PYTHONPATH=/app/TrajectoryAidedLearning

COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh
# Normalize line endings to avoid /dev/null$'\r' issues
RUN sed -i 's/\r$//' /app/docker-entrypoint.sh

EXPOSE 6080
CMD ["/app/docker-entrypoint.sh"]
