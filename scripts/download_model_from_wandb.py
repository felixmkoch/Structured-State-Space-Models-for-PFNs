import wandb
import torch


wandb_project = "mamba_project"
wandb_job_type = "download_mamba_model"
wandb_run_name = "Download Mamba Model"

MODEL_NAME = "mamba_150e"
alias = "latest"

model_name = f"{MODEL_NAME}:{alias}"

wandb_run = wandb.init(project=wandb_project, job_type=wandb_job_type, name=wandb_run_name)

artifact = wandb_run.use_artifact(model_name, type="model")

print(f"Download Model {MODEL_NAME} to wandb")

artifact_dir = artifact.download(root=".")

wandb.finish()