FROM tensorflow/tensorflow:2.11.0-gpu

WORKDIR /challenge

# Install required packages
RUN pip install numpy pandas scikit-learn wfdb matplotlib tensorflow==2.11.0

# Copy code files
COPY team_code.py /challenge/
COPY train_model.py /challenge/
COPY run_model.py /challenge/
COPY helper_code.py /challenge/
COPY requirements.txt /challenge/

# Health check to verify that the container is ready
HEALTHCHECK --interval=5m --timeout=3s \
  CMD python -c "import tensorflow as tf; print(tf.__version__)" || exit 1
