FROM public.ecr.aws/amazonlinux/amazonlinux:2023-minimal
COPY server /usr/local/bin/server
ENV PLATFORM=fargate-s3 KEV_LOCAL_DIR=/tmp
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/server"]
