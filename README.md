# Military-DS-Pipeline
To build the image run the following code
```
docker buildx build --push --platform=linux/arm64 -t dishimwe17/military-ds-pipeline:arm64 .
```

To run the image use
```
docker run --gpus all --name=military_ds_pipeline military_ds_pipeline:arm64
```