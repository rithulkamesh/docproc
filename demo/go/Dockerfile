# demo Go API + worker
FROM golang:1.23-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 go build -o /demo .

FROM alpine:3.19

RUN adduser -D -g "" app
USER app

WORKDIR /app
COPY --from=builder /demo .

EXPOSE 8080

# Default: run API. Override with ["/app/demo", "--worker"] for worker.
CMD ["/app/demo"]
