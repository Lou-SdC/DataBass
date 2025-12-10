# Python and pip image
# todo change to 3.14.2
FROM python:3.10-slim

# Set working directory
WORKDIR /databass

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8080

# go to the databass directory
WORKDIR /databass/databass

# Command to run the FastAPI application with Uvicorn
CMD ["uvicorn", "api.fast:app", "--host", "0.0.0.0", "--port", "8080"]
