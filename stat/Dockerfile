FROM golang:1.24-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY main.go .

RUN CGO_ENABLED=0 go build -o stat .

FROM alpine:3.21

WORKDIR /app

COPY --from=builder /app/stat .

EXPOSE 8002

CMD ["./stat"]
