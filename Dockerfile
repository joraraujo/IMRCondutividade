# Use a imagem oficial do Python
FROM python:3.10-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos de dependências
COPY requirements.txt ./
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt


# Copia o restante da aplicação
COPY . .

# Expõe a nova porta
EXPOSE 8503

# Comando para rodar o Streamlit na porta 8503
CMD ["streamlit", "run", "app.py", "--server.port=8503", "--server.address=0.0.0.0"]
