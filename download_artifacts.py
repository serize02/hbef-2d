import wandb

if __name__ == "__main__":
    run = wandb.init(project='echonet-dl')
    run.use_artifact('ernestoserize-constructor-university/echonet-dl/resnet50-unet:v1', type='model').download()
    run.use_artifact('ernestoserize-constructor-university/echonet-dl/error-predictor:v4', type='model').download()
    wandb.finish()