FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

# Start FastAPI first, wait 5s for it to be ready, then start Streamlit
CMD ["sh", "-c", "python run.py & sleep 5 && streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
