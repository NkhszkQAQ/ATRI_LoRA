FROM node:22-bookworm-slim AS build
WORKDIR /web
COPY webapp/upstream/web/package*.json ./
RUN npm install
COPY webapp/upstream/web/ ./
ARG VITE_API_BASE=/api
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

FROM nginx:1.27-alpine
COPY misc/docker/web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /web/dist /usr/share/nginx/html
EXPOSE 80
