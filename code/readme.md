# Code of masterthesis "Modeling object localization uncertainty for robot grasping"
In this reposetory, you can find all the code used for my thesis "Modeling object localization uncertainty for robot grasping".
In this file I will guide you through the different parts of the repository.
I will do this by going over each directory and explaining what it was/is used for and how to use it yourself.

All commands below are for linux-machines, however, there are very similar commands for windows and mac. Please also note that you should have `Python 3.12.3` and `pip 24.0` aleardy installed in your machine.

## Root
In the root directory (where this `readme.md` file is located), you find `requirements.txt` which holds all the libraries used throughout this repository.

To use the code effectively, first navigate to the root directory `code_thesis-manama`.
After which you should create a virtual environement by excecuting the following command.

```bash
python3 -m venv venv
```

Next, activate it by running the following.

```bash
source venv/bin/activate
```

After doing so, load the required libraries by running the following command.

```bash
pip install -r requirements.txt
```

You should now be all set up to run everything!
Note that everything should be run from this root directory using the format below.

```bash
python3 -m <subdirectory_example>.<example.py>
```

## /camera
This directory contains everything related to the Intel RealSense camera used to capture data.

### Camera.py
All the functionality to interact with the camera can be found in this file. 
This includes the calibration using the cv2 chesboard, the pixel to real-world coordinates mapping and functions to get and save frames.

### config.py
This file holds all the parameters related to this directory that need changing.

### real_world_mapping.py
This file contains the functionality to manually create the virutal grid.

To run it, execute the following command from the root directory:
```bash
python3 -m camera.real_world_mapping.py
```

### run_camera.py
This is a dummy script that lets you see if the camera works.

To run it, execute the following command from the root directory:
```bash
python3 -m camera.run_camera
```

### calibration_map.json
This file contains all the info needed for the pixel to real-world coordinates transformation used in `Camera.py`.

### checkerboard_pattern.png
This is an image of the checkerboard used for calibration.

## /data
This directory contains the dataset we created.

### regression
This holds the data used for training and validation.
It also holds a test set containing similar conditions to the training and validation set.

### out-of-distribution
This directory holds the data which are different from the training and validation. 
They include multiple objects, difficult lighting and human interference.

## /dataset_creation
This directory contains everything to do with creating the dataset.

### calculate_real_world_coordinates.py
This file has a function to do the translation from pixel to real-world coordinates.

### config.py
This file contains the variables that need changing for this directory.

### main_regression.py
This script was used to collect the data from the `/data` directory.
It gives the user an easy interface to annotate the captured data.

To run it, execute the following command from the root directory:

```bash
python3 -m dataset_creation.main_regression
```

## /models_to_test
This directory contains everything to do with the models and their training.

### ensemble
This directory contains the code for the custom CNN ensemble.

#### config.py
In this file, you can tweak the parameters of the model being trained.

#### ensemble.py
In this file, the classes for the individual models as well as the ensemble model are defined.

#### setup_training.py
This file actually trains the ensemble of custom CNNs using the parameters from `config.py`.
The models (individual and ensemble) get saved in the `models` directory.
The results on the validation set get saved in the `results` directory.

To run it, execute the following command from the root directory:

```bash
python3 -m models_to_test.ensemble.setup_training
```

#### final_models
This directory holds the config files for the models used in the thesis.
It also holds the results on the validation and test set.
The ensemble model's file (`.pth`-file) was too big to put on github and can be found using the google drive link at the end of this readme.

### MC_dropout
This directory is structured in the exact same way as the ensemble.

To train a MC dropout model, execute the following command from the root directory:

```bash
python3 -m models_to_test.MC_dropout.setup_training
```

### resnet
This directory is structured in the exact same way as the ensemble.

To train a resnet ensemble, execute the following command from the root directory:

```bash
python3 -m models_to_test.resnet.setup_training
```

### custom_training
This directory contains the class for the NLL-loss function used during the training of the models.

### convert_weights_to_model
This directory exists to convert the model weights, saved by the different models (ensemble, MC dropout or resnet), into python-objects that can easily be loaded.
Use the config file to (un)comment the correct parts.

To convert a model's weights, execute the following command from the root directory:
```bash
python3 -m models_to_test.convert_weights_to_model.convert_weights_to_model
```

## /robot_demonstrator
This directory contains a link to the github used to control the robot.

## /testing
This directory holds everything to do with the testing of the models.

### config.py
This contains all the variables that need changing for this directory.

### test_model.py
This tests a model on a given testset and reports its findings regarding loss and predictions.
The predictions are saved (by default) in `results/models_to_test/jsons`, while the loss gets printed to the teminal.

To run this, execute the following command from the root directory:

```bash
python3 -m testing.test_model.py
```

### test_statistics.py
This file tests a model using the classification metric (and thus using the uncertainty intervals).
It also generates the error-uncertainty graphs.
It prints its results to the terminal, while the graph gets saved (by default) in `results/models_to_test/graphs`.

### /models_to_test
This directory contains the models in `.pth` files (which need to be python objects), as well as the results of the scripts.

## /torch_dataset
This folder contains everything that has to do with the PyTorch side of the dataset as well as the normalisation.

### calculate_bounds.py
This scripts creates a `.json` file which contains the normalisation parameters.

To run this, execute the following command from the root directory:

```bash
python3 -m torch_dataset.calculate_bounds
```

### custom_dataset.py
This file contains the functionality to create a PyTorch compatible dataset from the dataset in the `/data` directory.
It is mostly used in the `/models_to_test` directory.

## Link to the google drive
This link contains the `.pth`-files of the used models:

https://drive.google.com/drive/folders/1hRC9rouUEC9QY4yYHB2osGcXyUrkcccM?usp=sharing
