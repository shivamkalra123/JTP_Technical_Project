# Stage 1: Build the Vite app
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build


# Stage 2: Serve using a lightweight web server (like nginx)
FROM nginx:stable-alpine

# Copy built assets from previous stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Optional: replace default nginx config with custom one (optional)
# COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
