# https://hub.docker.com/r/alpine/curl
# https://blog.thesparktree.com/cron-in-docker
FROM alpine/curl:3.14

USER root

RUN which crond && \
    rm -rf /etc/periodic

RUN apk add --update curl && \
    rm -rf /var/cache/apk/*

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# source: `docker run --rm -it alpine  crond -h`
# -f | Foreground
# -l N | Set log level. Most verbose 0, default 8
CMD ["crond", "-f", "-l", "2"]