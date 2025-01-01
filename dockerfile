# Use the latest Ubuntu image
FROM python:3.10


# Set working directory
WORKDIR /faceportal

RUN apt-get update && \
    apt-get install -y nginx zip nano bash curl net-tools iputils-ping libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/

# Copy necessary files into Docker container
COPY faceportal.py /faceportal
COPY requirements.txt /faceportal
COPY default /faceportal
COPY start.sh /faceportal
COPY DT_Body.pkl /faceportal
COPY RF_Body.pkl /faceportal
COPY SVM_Body.pkl /faceportal
COPY MLP_Body.pkl /faceportal
COPY DT_Face.pkl /faceportal
COPY RF_Face.pkl /faceportal
COPY SVM_Face.pkl /faceportal
COPY MLP_Face.pkl /faceportal
COPY FER_DNN.keras /faceportal

RUN mkdir -p /faceportal/Fair_Results && chmod 755 /faceportal/Fair_Results
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# Grant execute permissions to the script
RUN chmod +x /faceportal/start.sh
# Run the script
CMD ["/bin/bash", "/faceportal/start.sh"]
