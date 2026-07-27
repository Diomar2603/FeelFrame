# syntax=docker/dockerfile:1

# ---------- Etapa 1: build dos assets estáticos com Vite ----------
FROM node:22-alpine AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Variáveis VITE_* são "queimadas" (embutidas) no bundle JS durante o build —
# o navegador não tem acesso a variáveis de ambiente do container em runtime,
# por isso precisam chegar aqui como build args (ver DEPLOY.md e o workflow
# de CI/CD, que passa esses valores via --build-arg).
ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

RUN npm run build

# ---------- Etapa 2: serve os arquivos estáticos com nginx ----------
FROM nginx:1.27-alpine

# Arquivos em /etc/nginx/templates/*.template são processados com envsubst
# pelo entrypoint padrão da imagem oficial do nginx antes de iniciar,
# substituindo ${PORT} pelo valor que o Cloud Run injeta no container.
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

ENV PORT=8080
EXPOSE 8080
