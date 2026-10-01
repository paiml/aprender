FROM public.ecr.aws/lambda/provided:al2023-arm64
COPY kev.gguf /opt/kev.gguf
COPY bootstrap /var/runtime/bootstrap
ENV KEV_GGUF=/opt/kev.gguf PLATFORM=lambda-baked RAYON_NUM_THREADS=6
CMD ["handler"]
