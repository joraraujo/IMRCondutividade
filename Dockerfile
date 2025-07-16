FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./

# Pula verificação SSL
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org --index-url http://pypi.org/simple -r requirements.txt

COPY . .

EXPOSE 8503

CMD ["streamlit", "run", "app.py", "--server.port=8503", "--server.address=0.0.0.0"]
