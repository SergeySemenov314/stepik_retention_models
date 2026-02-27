FROM node:20-alpine

WORKDIR /app

COPY backend/package.json backend/package-lock.json* ./
RUN npm install --production

COPY backend/src/ ./src/
COPY backend/data/ ./data/

EXPOSE 3002

CMD ["node", "src/index.js"]
