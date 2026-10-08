FROM golang:1.25.6 AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config libopus-dev libopusfile-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY media-edge/go.mod media-edge/go.sum ./media-edge/
WORKDIR /src/media-edge
RUN go mod download
COPY media-edge/ ./
RUN CGO_ENABLED=1 go build -trimpath -ldflags='-s -w' -o /out/media-edge ./cmd/media-edge

FROM debian:bookworm-slim
ARG ORT_VERSION=1.29.0
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libopus0 libopusfile0 \
    && mkdir -p /usr/local/lib/onnxruntime \
    && tmp="$(mktemp -d)" \
    && curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" -o "$tmp/ort.tgz" \
    && tar -xzf "$tmp/ort.tgz" -C "$tmp" \
    && cp "$tmp/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so"* /usr/local/lib/onnxruntime/ \
    && rm -rf "$tmp" \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
COPY --from=build /out/media-edge /usr/local/bin/media-edge
WORKDIR /app
RUN mkdir -p /var/lib/media-edge/certs && chown -R app:app /var/lib/media-edge
USER app
EXPOSE 8443 9101
ENTRYPOINT ["/usr/local/bin/media-edge"]
