# PyTorch Wine Lambda - Container Image
# For scientific comparison with .apr format

FROM public.ecr.aws/lambda/python:3.11

# Install PyTorch CPU-only (smaller image)
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Copy function code with proper permissions
COPY --chmod=644 lambda_function.py ${LAMBDA_TASK_ROOT}/

# Set handler
CMD ["lambda_function.lambda_handler"]
