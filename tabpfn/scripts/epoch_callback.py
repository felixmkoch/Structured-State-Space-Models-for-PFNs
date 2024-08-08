from tabpfn.scripts.model_builder import get_model, save_model

def epoch_callback(model, epoch, config, model_name):

    if epoch % 10 != 0: return

    config["stop_epoch"] = epoch

    print("Saving model via epoch callback ...")

    save_model(
        model=model,
        path="./",
        filename=f'tabpfn/models_diff/callback_{model_name}.cpkt',
        config_sample=config
    )

