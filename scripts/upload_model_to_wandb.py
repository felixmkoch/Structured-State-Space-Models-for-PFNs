import wandb
import torch


wandb_project = "mamba_project"
wandb_job_type = "upload_mamba_model"
wandb_run_name = "Upload Mamba Model"

MODEL_NAME = "mamba_150e"
wandb_run = wandb.init(project=wandb_project, job_type=wandb_job_type, name=wandb_run_name)

MODEL_PATH = "mamba_quick_model.cpkt"

artifact = wandb.Artifact(MODEL_NAME, type="model")

artifact.add_file(MODEL_PATH)

print(f"Upload Model {MODEL_NAME} to wandb")

wandb_run.log_artifact(artifact)

wandb.finish()