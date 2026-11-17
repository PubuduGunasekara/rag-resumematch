# Next.js site. The browser talks only to this server; /api/* is proxied to
# the Python API at API_URL (see next.config.mjs), so the API needs no
# public port and no CORS.
FROM node:22-alpine AS build
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# The proxy rule is written into the build, so the API address is a build
# argument: http://api:8000 is the API service on the compose network.
ARG API_URL=http://api:8000
ENV API_URL=$API_URL NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:22-alpine
WORKDIR /web
ARG API_URL=http://api:8000
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 API_URL=$API_URL
COPY --from=build /web/package.json /web/package-lock.json ./
RUN npm ci --omit=dev
COPY --from=build /web/.next ./.next
COPY --from=build /web/next.config.mjs ./
EXPOSE 3100
CMD ["npm", "run", "start"]
