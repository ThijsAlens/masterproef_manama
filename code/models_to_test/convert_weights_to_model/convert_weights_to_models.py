import torch

import models_to_test.convert_weights_to_model.config as config


if __name__ == "__main__":
    
    model_weights = torch.load(config.PATH_TO_MODEL_WEIGHTS)
    
    if config.MC_DROPOUT:
        model = config.ENSEMBLE_CLASS(config.INDIVIDUAL_MODEL_CLASS(**config.MODEL_PARAMETERS), n_samples=config.NUMBER_OF_SAMPLES)
    else:
        empty_models = []
        for _ in range(config.NUMBER_OF_MODELS):
            model = config.INDIVIDUAL_MODEL_CLASS(**config.MODEL_PARAMETERS)
            empty_models.append(model)

        model = config.ENSEMBLE_CLASS(empty_models)
        
    model.load_state_dict(model_weights)
    torch.save(model, config.PATH_TO_MODEL)