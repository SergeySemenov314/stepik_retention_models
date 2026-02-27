FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install --production

COPY src/ ./src/
COPY data/ ./data/

EXPOSE 3002

CMD ["node", "src/index.js"]
