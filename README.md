# Enhancing Image Retrieval Performance with Generative Models in Siamese Networks
The official implementation of [Enhancing Image Retrieval Performance with Generative Models in Siamese Networks](https://scholar.google.com/)  
[Alejandro Golfe](https://scholar.google.com/), [Adrián Colomer](https://scholar.google.com/](https://scholar.google.es/citations?user=U6BEiIEAAAAJ&hl=es&oi=ao), [José Prades](https://www.upv.es/ficha-personal/jprades), [Valery Naranjo](https://scholar.google.es/citations?user=jk4XsG0AAAAJ&hl=es&oi=ao)  
| [Paper](URL_AQUI) | [Code](https://github.com/cvblab/CBIR_SYNTHETIC_VIEWS) |  

![Model Diagram](images/main.pdf)

## Overview

Prostate cancer is a critical healthcare challenge globally and is one of the most prevalent types of cancer in men. Early and accurate diagnosis is essential for effective treatment and improved patient outcomes. 

In the existing literature, computer-aided diagnosis (CAD) solutions have been developed to assist pathologists in various tasks, including classification, diagnosis, and prostate cancer grading. Content-based image retrieval (CBIR) techniques provide valuable approaches to enhance these computer-aided solutions.

This study evaluates how generative deep learning models can improve the quality of retrievals within a CBIR system. Specifically, we propose applying a Siamese Network approach, which enables us to learn how to encode image patches into latent representations for retrieval purposes. We used the ProGleason-GAN framework trained on the SiCAPv2 dataset to create similar pairs of input patches. Our observations indicate that introducing synthetic patches leads to notable improvements in the evaluated metrics, underscoring the utility of generative models within CBIR tasks. 

Furthermore, this work is the first in the literature where latent representations optimized for CBIR are used to train an attention mechanism for performing Gleason Scoring of a Whole Slide Image (WSI).


## Usage 

1. Clone this repository:  

   ```sh
   git clone https://github.com/cvblab/CBIR_SYNTHETIC_VIEWS
   cd nombre_del_repositorio

  
2. Install dependencies:

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
3. File Structure

The main components of this project are:

- `model.py`: Contains the definition of the neural network architecture.
- `evaluation.py`: Contains evaluation functions.
- `utils.py`: Contains utility functions like data loading, transformations, etc.
- `config.ini`: Configuration file containing all the parameters required for training and evaluation.
- `train.py`: Main script for training and evaluation, including argument parsing.

4. Configuration

Before running the code, you must configure the `config.ini` file to set paths and parameters.

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

5. Running the Code

To train the model and evaluate its performance, run the following command in the terminal:

```bash
python train.py --path_experiment /path/to/experiment
```

The script will use the configurations set in `config.ini` to perform training and evaluation.

6. Arguments

- `--path_experiment`: The path to the directory containing the experiment's configuration files and where the results will be stored.
- `--last_epoch`: Optional argument to specify the last trained epoch (default is `0` to start from scratch).

Example:

```bash
python train.py --path_experiment /path/to/experiment --last_epoch 50
```

This command will resume training from epoch 50.

## Acknowledgment
This repository is mainly based on [SimCLR](https://github.com/sthalles/SimCLR) code base. We sincerely thank prior authors on this topic for their code base.

## Citation

If you use this code in your research, please cite it appropriately.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
