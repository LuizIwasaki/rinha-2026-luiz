# Stage 1: Build
FROM alpine:3.19 AS builder
RUN apk add --no-cache build-base cmake linux-headers
WORKDIR /app
COPY . .
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j$(nproc)

# Stage 2: Preprocess (k-means IVF index)
FROM alpine:3.19 AS preprocessor
RUN apk add --no-cache libstdc++ gzip
WORKDIR /data
COPY --from=builder /app/build/preprocess /usr/local/bin/preprocess
COPY references.json.gz /tmp/references.json.gz
RUN gunzip -k /tmp/references.json.gz && \
    preprocess /tmp/references.json /data/vectors.bin /data/labels.bin /data/centroids.bin /data/offsets.bin && \
    rm /tmp/references.json /tmp/references.json.gz

# Stage 3: Runtime
FROM alpine:3.19
RUN apk add --no-cache libstdc++
WORKDIR /app
COPY --from=builder /app/build/rinha_api ./rinha_api
COPY --from=preprocessor /data/vectors.bin /data/vectors.bin
COPY --from=preprocessor /data/labels.bin /data/labels.bin
COPY --from=preprocessor /data/centroids.bin /data/centroids.bin
COPY --from=preprocessor /data/offsets.bin /data/offsets.bin
ENV DATA_DIR=/data
EXPOSE 8080
CMD ["./rinha_api"]
