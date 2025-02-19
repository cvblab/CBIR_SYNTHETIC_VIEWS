 Model Training and Evaluation

This project trains and evaluates a machine learning model on a dataset using a synthetic data generation technique. The code supports training with a custom model, evaluation using k-NN, and saves the obtained results.

## Requirements

Ensure that the following dependencies are installed:

- Python 3.x
- PyTorch
- tqdm
- thop
- pandas
- matplotlib
- seaborn
- configparser
- ast
- ssl

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## File Structure

The main components of this project are:

- `model.py`: Contains the definition of the neural network architecture.
- `evaluation.py`: Contains evaluation functions.
- `utils.py`: Contains utility functions like data loading, transformations, etc.
- `config.ini`: Configuration file containing all the parameters required for training and evaluation.
- `train.py`: Main script for training and evaluation, including argument parsing.

## Configuration

Before running the code, you will need to configure the `config.ini` file to set paths and parameters.

Example `config.ini` file format:

```ini
[VARIABLES]
path_csv_sicap_test = /path/to/test.csv
path_sicap_images = /path/to/images
path_sicap_csv = /path/to/dataset.csv
channels_img = 3
num_workers = 4
results_path = /path/to/results
evaluation_path = /path/to/evaluation
progleason_model = /path/to/progleason/model.pth
z_dim = 128
noise_dim = 100
temperature = 0.07
k = 5
batch_size = 32
epochs = 100
subset = [1, 2, 3]
seed = 42
lr = 0.001
```

Make sure that the paths to your datasets and models are correctly specified.

## Running the Code

To train the model and evaluate its performance, run the following command in the terminal:

```bash
python train.py --path_experiment /path/to/experiment
```

The script will use the configurations set in `config.ini` to perform training and evaluation.

### Arguments

- `--path_experiment`: The path to the directory containing the experiment's configuration files and where the results will be stored.
- `--last_epoch`: Optional argument to specify the last trained epoch (default is `0` to start from scratch).

Example:

```bash
python train.py --path_experiment /path/to/experiment --last_epoch 50
```

This command will resume training from epoch 50.

## Model Architecture

The model used in this project is based on a custom architecture defined in `model.py`. It is designed for learning a representation from images and utilizes synthetic data generation techniques for training.

## Training Process

During training, the model will be optimized using the Adam optimizer. The synthetic data generator is used to create augmented data samples, which are then used to compute the loss and update the model. The training process runs for a specified number of epochs, and the loss is reported after each epoch.

## Evaluation

The model is evaluated using a k-NN approach on the test set. During evaluation, the top-1 and top-5 accuracy metrics are computed to measure the model's performance. The evaluation results are logged and saved.

## Results and Checkpoints

Training progress, including training loss and test accuracies (top-1 and top-5), will be saved in the `results_path` specified in the `config.ini` file. The model's weights will also be saved if it achieves a new best top-1 accuracy.

## Citation

If you use this code in your research, please cite it appropriately.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
