# Estágio de Compilação
FROM alpine:3.19 AS builder

RUN apk add --no-cache build-base cmake linux-headers

WORKDIR /app
COPY . .

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --config Release

# Estágio Final (Imagem Minúscula)
FROM alpine:3.19

# Adiciona a libstdc++ que é necessária para C++
RUN apk add --no-cache libstdc++

WORKDIR /app
COPY --from=builder /app/build/rinha_api .

EXPOSE 8080

CMD ["./rinha_api"]
