FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# RUN pip install -U crawl4ai
# RUN /usr/local/bin/crawl4ai-setup

COPY . .

RUN chmod +x start.sh

CMD ["./start.sh"]
