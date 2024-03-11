# Step 1: Base Image
FROM python:3.8-slim

# Step 2: Working Directory
WORKDIR /app

# Step 3: Environment Variables
# Example: ENV VARIABLE_NAME=value
# Use --env-file or orchestration tools for sensitive keys

# Step 4: Dependencies
# System dependencies (if any)
# RUN apt-get update && apt-get install -y <package-name>

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy Application
COPY . .

# Step 6: Expose Port
# Example for a web application
# EXPOSE 8000

# Step 7: Command to Run the Application
CMD ["python", "app.py"]