# Use an official Python runtime as a base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH to ensure src module is found
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Run the training script (change this to the script you want to execute)
CMD ["python", "src/model/train_model.py"]
