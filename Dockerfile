FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-8
# FROM asia-docker.pkg.dev/vertex-ai/training/tf-gpu.2-11.py310:latest

WORKDIR /

# Copies the trainer code to the docker image.
COPY trainer /trainer

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.train"]