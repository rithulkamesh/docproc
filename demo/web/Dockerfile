FROM node:22-alpine AS builder

WORKDIR /app

# API URL for browser requests (default: same host, port 8080)
ARG VITE_DOCPROC_API_URL=http://localhost:8080
ENV VITE_DOCPROC_API_URL=$VITE_DOCPROC_API_URL

COPY package.json package-lock.json* ./
RUN npm install

COPY . .
RUN npm run build

FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

