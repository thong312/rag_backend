# -----------------------------
# Dockerfile tối ưu cho RAG app
# -----------------------------

# 1️⃣ Base image nhẹ
FROM python:3.12-slim

# 2️⃣ Thiết lập thư mục làm việc
WORKDIR /app

# 3️⃣ Cài system dependencies cần thiết cho ML/AI packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev libffi-dev libssl-dev curl git \
    && apt-get purge -y git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*


# 4️⃣ Copy requirements và upgrade pip trước
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


# 5️⃣ Copy toàn bộ source code
COPY . .

# 6️⃣ Expose port nếu cần (FastAPI)
EXPOSE 8080

# 7️⃣ Run app
CMD ["python", "app2.py"]
