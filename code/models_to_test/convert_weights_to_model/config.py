import models_to_test.ensemble.ensemble as ensemble
import models_to_test.resnet.resnet as resnet
import models_to_test.MC_dropout.mc_dropout as mc_dropout


PATH_TO_MODEL_WEIGHTS = "testing/model_tests/MC_dropout_4.pth"
PATH_TO_MODEL = "testing/model_tests/MC_dropout_4.pth"


# ------------------------------------------------------
# CUSTOM ENSEMBLE (uncomment and change the necessary parameters to convert this model)
# ------------------------------------------------------

# ENSEMBLE_CLASS = ensemble.DeepEnsemble
# INDIVIDUAL_MODEL_CLASS = ensemble.SimpleCNNRegressionModel
# NUMBER_OF_MODELS = 3
# MODEL_PARAMETERS = {
#     "input_dims": (3, 370, 250),
#     "hidden_channels": [8, 16, 32],
#     "output_size": 2 # (x, y)
# }

# MC_DROPOUT = False # Do not touch

# ------------------------------------------------------
# RESNET ENSEMBLE (uncomment and change the necessary parameters to convert this model)
# ------------------------------------------------------

# ENSEMBLE_CLASS = resnet.DeepEnsemble
# INDIVIDUAL_MODEL_CLASS = resnet.ResNetRegressionModel
# NUMBER_OF_MODELS = 5
# MODEL_PARAMETERS = {
#     "input_dims": (3, 370, 250),
#     "output_size": 2, # (x, y)
#     "freeze_backbone": True,
#     "number_of_res_blocks": 4
# }

# MC_DROPOUT = False # Do not touch

# ------------------------------------------------------
# MC DROPOUT (uncomment and change the necessary parameters to convert this model)
# ------------------------------------------------------

ENSEMBLE_CLASS = mc_dropout.MC_Dropout
INDIVIDUAL_MODEL_CLASS = mc_dropout.SimpleCNNRegressionModelDropout
NUMBER_OF_SAMPLES = 100
MODEL_PARAMETERS = {
    "input_dims": (3, 370, 250),
    "hidden_channels": [32, 64, 128],
    "output_size": 2,
    "p_list": [0.1, 0.1, 0.1]
}

MC_DROPOUT = True # Do not touch