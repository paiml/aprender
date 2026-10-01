FROM public.ecr.aws/amazonlinux/amazonlinux:2023-minimal
COPY kev.gguf /opt/kev.gguf
COPY server /usr/local/bin/server
ENV KEV_GGUF=/opt/kev.gguf PLATFORM=fargate-baked
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/server"]
