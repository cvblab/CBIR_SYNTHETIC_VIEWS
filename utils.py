from PIL import Image
from torchvision import transforms
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
import torchvision.transforms.functional as F_transform
from models_Progan_ACGAN import Generator
import model
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# Transformation to be applied on training images
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Crop and resize the image to 224x224
    transforms.RandomHorizontalFlip(p=0.5),  # Apply horizontal flip with 50% probability
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    # Apply color jitter with 80% probability
    transforms.RandomGrayscale(p=0.2),  # Randomly convert the image to grayscale with 20% probability
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  # Normalize with specific mean and std
])

# Transformation to be applied on test images
test_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  # Normalize
])


class SiCAPv2(Dataset):
    """
    Custom dataset for the SiCAPv2 dataset.

    Arguments:
        csv_file (str): Path to the CSV file containing image metadata (paths and labels).
        root_dir (str): Directory where images are stored.
        transform (callable, optional): A function/transform to apply on the images. Defaults to `train_transform`.
        shape (tuple, optional): Desired shape of the images. Defaults to (512, 512).
        subset (bool, optional): Whether to load only a subset of the dataset. Defaults to False.

    Attributes:
        image_names (list): List of image filenames.
        labels (list): List of image labels.
        pseudolabels (list): Pseudo labels for images (currently unused).
        root_dir (str): Root directory where images are stored.
        transform (callable): Transformation function to be applied on images.
        dataframe (DataFrame): Pandas dataframe containing image metadata.
        shape (tuple): Shape of images.
        subset (bool): Flag indicating whether to load a subset.
        image_list (list): List of image tensors after transformations.
        labels_list (list): List of corresponding labels for images.
        path_list (list): List of image file paths.
    """

    def transform_labels(self, dataframe: pd.DataFrame, dictionary_classes: dict) -> tuple:
        """
        Transforms the labels from one-hot encoded to class indices.

        Arguments:
            dataframe (pd.DataFrame): DataFrame containing image metadata and one-hot encoded labels.
            dictionary_classes (dict): Dictionary mapping class names to class indices.

        Returns:
            tuple: A tuple containing:
                - List of image file paths (X).
                - List of class labels corresponding to images (y).
                - List of pseudolabels (currently unused, returns an empty list).
        """
        X = []  # Image paths
        y = []  # Image labels
        pseudolabels = []  # Empty list, not used
        for i in range(len(dataframe)):
            X.append(dataframe.loc[i][0])  # Add image path
            # Check the one-hot encoding and assign the appropriate class
            if (dataframe.loc[i][1] == 1):
                y.append(dictionary_classes['NC'])  # Class 'NC' mapped to 0
            elif (dataframe.loc[i][2] == 1):
                y.append(dictionary_classes['G3'])  # Class 'G3' mapped to 1
            elif (dataframe.loc[i][3] == 1):
                y.append(dictionary_classes['G4'])  # Class 'G4' mapped to 2
            elif (dataframe.loc[i][4] == 1):
                y.append(dictionary_classes['G5'])  # Class 'G5' mapped to 3
            elif (dataframe.loc[i][5] == 1):
                y.append(dictionary_classes['G4C'])  # Class 'G4C' mapped to 2 (same as 'G4')

        return np.array(X), np.array(y), np.array(pseudolabels)

    def __init__(self, csv_file: str, root_dir: str, transform: callable = train_transform, shape: tuple = (512, 512),
                 subset: bool = False):
        """
        Initializes the SiCAPv2 dataset by loading images, labels, and applying transformations.

        Arguments:
            csv_file (str): Path to the CSV file containing image metadata.
            root_dir (str): Root directory where images are stored.
            transform (callable, optional): A function/transform to apply on the images. Defaults to `train_transform`.
            shape (tuple, optional): Desired shape of the images. Defaults to (512, 512).
            subset (bool, optional): Whether to load only a subset of the dataset. Defaults to False.
        """
        self.dictionary = {'NC': 0, 'G3': 1, 'G4': 2, 'G5': 3, 'G4C': 2}  # Mapping of class names to indices
        dataframe = pd.read_excel(csv_file)  # Load metadata from the Excel file
        self.image_names, self.labels, self.pseudolabels = self.transform_labels(dataframe,
                                                                                 self.dictionary)  # Get image names and labels
        self.root_dir = root_dir  # Set the root directory for images
        self.transform = transform  # Set the transformation function to apply on images
        self.dataframe = dataframe  # Store the entire dataframe
        self.shape = shape  # Store the shape
        self.subset = subset  # Store subset flag
        self.image_list = []  # List to store image tensors
        self.labels_list = []  # List to store image labels
        self.path_list = []  # List to store image file paths

        resize_and_to_tensor = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
        ])

        if self.subset:
            self.image_names = self.image_names[:100]  # Only take the first 100 images if subset flag is True

        print('Loading SiCAPv2 images into memory...')
        for position, element in enumerate(tqdm(self.image_names)):
            img_path = os.path.join(self.root_dir, element)  # Get the full image path
            image = Image.open(img_path)  # Open image using PIL
            image = np.array(image)  # Convert the image to a NumPy array
            image = Image.fromarray(image, 'RGB')  # Convert back to PIL image in RGB format
            image = resize_and_to_tensor(image)  # Apply resizing transformation
            y_label = torch.tensor(self.labels[position])  # Get the label for the image
            self.image_list.append(image)  # Add image tensor to list
            self.labels_list.append(y_label)  # Add label to list
            self.path_list.append(os.path.join(self.root_dir, element))  # Add image path to list

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The total number of images in the dataset (or subset).
        """
        if self.subset:
            return 100  # If subset is True, return only 100 images
        else:
            return len(self.dataframe)  # Otherwise, return the full dataset length

    def __getitem__(self, item: int) -> tuple:
        """
        Returns a tuple containing a pair of augmented images, label, and image path at the specified index.

        Arguments:
            item (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing:
                - image1 (Tensor): Augmented version of the first image.
                - image2 (Tensor): Augmented version of the second image.
                - y_label (Tensor): Label for the image.
                - path (str): Path to the image.
        """
        image1 = self.transform(self.image_list[item])  # Apply transformation to the first image
        image2 = self.transform(self.image_list[item])  # Apply transformation to the second image (same image)
        y_label = self.labels_list[item]  # Get the label for the image
        path = self.path_list[item]  # Get the image path

        return image1, image2, y_label, path


def tsne_representations(train_loader: torch.utils.data.DataLoader,
                         encoder: torch.nn.Module,
                         epoch: int,
                         results_path: str) -> None:
    """
    Generates 2D and 3D t-SNE visualizations of the latent space representations for the training data.

    Arguments:
        train_loader (torch.utils.data.DataLoader): DataLoader for loading training data in batches.
        encoder (torch.nn.Module): The model (encoder) used to generate latent representations.
        epoch (int): Current epoch number used for saving the output images in a folder specific to the epoch.
        results_path (str): Path to save the generated t-SNE plots.

    Returns:
        None: The function saves the generated t-SNE plots as PNG images.
    """
    # Create directories for saving the results (if they don't exist)
    if not os.path.exists(os.path.join(results_path, 'representations', str(epoch))):
        os.makedirs(os.path.join(results_path, 'representations', str(epoch)))

    latent_representations = []  # List to store latent representations
    label_list = []  # List to store the corresponding labels
    print('Creating latent representations')
    encoder.eval()  # Set the encoder in evaluation mode
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(train_loader, leave=True)):
            try:
                patches, _, labels, _, _ = data
                patches = patches.cuda()  # Move data to GPU if available
                feature, out = encoder(patches)  # Forward pass through the encoder
                latent_representations.append(out.cpu().detach())  # Store the output
                label_list.append(labels.cpu().detach())  # Store the labels
            except:
                patches, _, labels, _, _ = data  # Handle any data loading issue

    # Concatenate latent representations and labels into tensors
    latents = torch.cat(latent_representations, dim=0)
    labels = np.concatenate(label_list, axis=0)

    # Apply t-SNE for dimensionality reduction (2D visualization)
    tsne = TSNE(n_components=2, perplexity=30.0, n_iter=1000, random_state=42)
    tsne_vectors = tsne.fit_transform(latents)

    # Plot the 2D t-SNE representation
    fig, ax = plt.subplots()
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta',
              'lightblue', 'lightgreen', 'salmon', 'gold', 'teal', 'lavender', 'lime', 'indigo', 'coral']

    for label in set(labels):  # Plot each class with a different color
        x = tsne_vectors[labels == label, 0]
        y = tsne_vectors[labels == label, 1]
        color = colors[label % len(colors)]  # Get color for current label
        plt.scatter(x, y, label=label, color=color)

    ax.legend()  # Add legend to the plot
    plt.show()
    ax.set_title('T-SNE')  # Set title
    fig.savefig(os.path.join(results_path, 'representations', str(epoch), 'tsne_2d.png'))  # Save the plot

    plt.clf()  # Clear the current figure

    # Apply t-SNE for 3D visualization
    tsne = TSNE(n_components=3, perplexity=30.0, n_iter=1000, random_state=42)
    tsne_vectors = tsne.fit_transform(latents)

    # Plot the 3D t-SNE representation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for label in set(labels):  # Plot each class with a different color
        x = tsne_vectors[labels == label, 0]
        y = tsne_vectors[labels == label, 1]
        z = tsne_vectors[labels == label, 2]
        ax.scatter(x, y, z, label=label)

    ax.legend()  # Add legend to the plot
    ax.set_title('T-SNE')  # Set title
    fig.savefig(os.path.join(results_path, 'representations', str(epoch), 'tsne_3d.png'))  # Save the plot
    plt.show()
    plt.clf()  # Clear the current figure


def topk_accuracy(N: np.ndarray, true_label: int, labels_memory_bank: list, k: int) -> int:
    """
    Computes top-k accuracy, checking if the true label is in the top-k predictions.

    Arguments:
        N (np.ndarray): Array of predicted distances or similarity scores.
        true_label (int): The true class label.
        labels_memory_bank (list): List of stored labels in the memory bank.
        k (int): The number of top predictions to consider.

    Returns:
        int: 1 if the true label is within the top-k predictions, 0 otherwise.
    """
    predictions = []
    N = N[:k]  # Select the top-k predictions
    for i, row in enumerate(N):
        idx_in_dataset = int(row[1])  # Get the index in the dataset
        predictions.append(labels_memory_bank[idx_in_dataset][0])  # Add the corresponding label to predictions

    correct = 0
    if true_label in predictions:  # Check if the true label is in the top-k predictions
        correct += 1

    return 1 if correct > 0 else 0  # Return 1 if correct, 0 if incorrect


def majority_vote(N: np.ndarray, true_label: int, labels_memory_bank: list, k: int) -> int:
    """
    Performs majority vote on the top-k predictions to decide the final prediction.

    Arguments:
        N (np.ndarray): Array of predicted distances or similarity scores.
        true_label (int): The true class label.
        labels_memory_bank (list): List of stored labels in the memory bank.
        k (int): The number of top predictions to consider.

    Returns:
        int: 1 if the majority vote matches the true label, 0 otherwise.
    """
    predictions = []
    true_label = int(true_label)
    N = N[:k]  # Select the top-k predictions
    for i, row in enumerate(N):
        idx_in_dataset = int(row[1])  # Get the index in the dataset
        predictions.append(labels_memory_bank[idx_in_dataset][0])  # Add the corresponding label to predictions

    # Perform majority vote
    counter = Counter(predictions)
    frequent_labels = counter.most_common()
    max_frequency = frequent_labels[0][1]  # Get the maximum frequency

    # Filter labels with maximum frequency
    majority_labels = [label for label, freq in frequent_labels if freq == max_frequency]

    # Take the mode label if there is a clear winner
    if len(majority_labels) == 1:
        mode_label = majority_labels[0]
    else:
        mode_label = min(majority_labels)  # Resolve tie by taking the smallest label

    # Check if the majority vote matches the true label
    if mode_label == true_label:
        prediction = 1  # Correct prediction
    else:
        prediction = 0  # Incorrect prediction

    return prediction


def plot_curves(df: pd.DataFrame, results_path: str) -> None:
    """
    Plots and saves accuracy curves from the provided DataFrame.

    Arguments:
        df (pd.DataFrame): DataFrame containing the accuracy metrics to plot.
        results_path (str): Path to save the resulting plots.

    Returns:
        None: The function saves the generated curves as PNG images.
    """
    if not os.path.exists(os.path.join(results_path, 'curves')):
        os.makedirs(os.path.join(results_path, 'curves'))  # Create directory for curves if it doesn't exist

    df_reset = df.reset_index(drop=True)  # Reset the index of the DataFrame

    # Columns to plot (Top-1 and Top-5 accuracies)
    columns_1 = ['test_acc@1', 'test_acc@5']

    plt.figure(figsize=(12, 6))  # Set figure size

    for column in columns_1:
        plt.plot(df_reset.index, df_reset[column], marker='o', linestyle='-', label=column)  # Plot each column

    plt.title('Top-k accuracy')  # Set plot title
    plt.xlabel('Index')  # Set x-axis label
    plt.ylabel('Value')  # Set y-axis label
    plt.grid(True)  # Add grid to the plot
    plt.legend()  # Add legend

    plt.tight_layout()  # Adjust layout to avoid overlapping
    plt.savefig(os.path.join(results_path, 'curves', 'combined_graphs.png'))  # Save the plot as a PNG file
    plt.show()  # Show the plot


def create_synthetic_similar_sample(
        batch_size: int,
        labels: torch.Tensor,
        progleason_model: torch.nn.Module,
        noise_dim: int,
        device: torch.device
) -> torch.Tensor:
    """
    Generates synthetic images similar to the given labels by feeding noise through a model.

    Args:
    - batch_size (int): The number of synthetic samples to generate in a single batch.
    - labels (torch.Tensor): The labels associated with each synthetic sample (shape: batch_size).
    - progleason_model (torch.nn.Module): The model used to generate synthetic images, given the noise and labels.
    - noise_dim (int): The dimension of the noise vector that will be passed through the model.
    - device (torch.device): The device (CPU or GPU) where the computations will take place.

    Returns:
    - torch.Tensor: A batch of synthetic images transformed into tensors, ready for further processing.
    """
    step = 6
    alpha = 1
    input_noise = torch.randn(batch_size, noise_dim, 1, 1).to(device, non_blocking=True)
    fake = progleason_model(input_noise, labels, alpha, step)

    image_size = 224
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize image to 224x224
        transforms.ToTensor(),  # Convert image to a tensor
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])  # Normalize

    batch_transformed = torch.stack([transform(F_transform.to_pil_image(img_tensor)) for img_tensor in fake], dim=0)
    batch_transformed = batch_transformed.to(device)
    return batch_transformed


def load_synthetic_model(name_model: str, device: torch.device) -> torch.nn.Module:
    """
    Loads a pre-trained synthetic image generator model from a checkpoint.

    Args:
    - name_model (str): Path to the pre-trained model checkpoint.
    - device (torch.device): The device (CPU or GPU) where the model will be loaded.

    Returns:
    - torch.nn.Module: The loaded model.
    """
    dimension = 512
    model = Generator(dimension, n_classes=4, in_channels=512, img_channels=3)
    model.to(device)

    checkpoint = torch.load(name_model, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # Set the model to evaluation mode
    return model


def load_pretrained_encoder(path_model: str, dimension: int) -> torch.nn.Module:
    """
    Loads a pre-trained encoder model from a checkpoint.

    Args:
    - path_model (str): Path to the pre-trained encoder model checkpoint.
    - dimension (int): The dimension of the encoded representation.

    Returns:
    - torch.nn.Module: The loaded encoder model.
    """
    encoder = model.Model(dimension)
    encoder.load_state_dict(torch.load(path_model), strict=False)
    encoder.cuda()
    encoder.eval()  # Set the model to evaluation mode
    return encoder


def create_database(
        output_path: str,
        output_path_total: str,
        encoder: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        device: torch.device
) -> None:
    """
    Extracts latent representations from a dataset and stores them in a memory bank.

    Args:
    - output_path (str): Path where intermediate memory bank files will be saved.
    - output_path_total (str): Path where the final concatenated memory bank will be saved.
    - encoder (torch.nn.Module): The encoder model used to extract latent representations.
    - train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
    - device (torch.device): The device (CPU or GPU) where the computations will take place.

    Returns:
    - None: This function saves the data as .npy files, so there is no return value.
    """
    loop = tqdm(train_loader, leave=True)
    with torch.no_grad():
        for batch_idx, data in enumerate(loop):
            encoded_list, label_list, path_list = [], [], []

            imgs, _, labels, paths = data
            imgs = imgs.to(device)
            _, e_img = encoder(imgs)

            for i, encoded_patch in enumerate(e_img):
                encoded_list.append(encoded_patch.cpu().detach().numpy())
                label_list.append(labels[i])
                path_list.append(paths[i])

            encoded_array = np.array(encoded_list)
            label_array = np.array(label_list)
            path_array = np.array(path_list)

            label_array = label_array.reshape((-1, 1))
            path_array = path_array.reshape((-1, 1))

            encoded_output_file = os.path.join(output_path, 'memory_bank_' + str(batch_idx) + '.npy')
            label_output_file = os.path.join(output_path, 'memory_bank_labels_' + str(batch_idx) + '.npy')
            path_output_file = os.path.join(output_path, 'memory_bank_paths_' + str(batch_idx) + '.npy')

            np.save(encoded_output_file, encoded_array)
            np.save(label_output_file, label_array)
            np.save(path_output_file, path_array)

    # Combine all saved .npy files into a single memory bank
    # This step ensures that the memory bank contains all the data from the dataset
    num_files = int(len(os.listdir(output_path)) / 3)
    for batch_idx in range(1, num_files):
        file1_2 = os.path.join(output_path, 'memory_bank_' + str(batch_idx) + '.npy')
        file2_2 = os.path.join(output_path, 'memory_bank_labels_' + str(batch_idx) + '.npy')
        file3_2 = os.path.join(output_path, 'memory_bank_paths_' + str(batch_idx) + '.npy')

        array1_2 = np.load(file1_2)
        array2_2 = np.load(file2_2)
        array3_2 = np.load(file3_2)

        array1 = np.concatenate((array1, array1_2), axis=0)
        array2 = np.concatenate((array2, array2_2), axis=0)
        array3 = np.concatenate((array3, array3_2), axis=0)

    # Save the final combined memory bank
    np.save(os.path.join(output_path_total, 'memory_bank_total.npy'), array1)
    np.save(os.path.join(output_path_total, 'memory_bank_labels_total.npy'), array2)
    np.save(os.path.join(output_path_total, 'memory_bank_paths_total.npy'), array3)


def compute_cosine_similarity(
        encoded_patch: torch.Tensor,
        memory_bank: np.ndarray,
        k: int
) -> np.ndarray:
    """
    Computes the cosine similarity between an encoded patch and a memory bank.

    Args:
    - encoded_patch (torch.Tensor): The latent representation of the query image (a 1D tensor).
    - memory_bank (np.ndarray): The array containing latent representations of the memory bank.
    - k (int): The number of most similar samples to retrieve.

    Returns:
    - np.ndarray: Indices of the top k most similar samples from the memory bank.
    """
    encoded_patch = encoded_patch.cpu().numpy().squeeze()
    cosine_similarities = cosine_similarity([encoded_patch], memory_bank)
    k_similar_indices = np.argsort(cosine_similarities[0])[-k:][::-1]
    return k_similar_indices


def topk_accuracy_cosine_similarity(
        k_similar_indices: np.ndarray,
        true_label: int,
        labels_memory_bank: np.ndarray,
        k: int
) -> Tuple[int, np.ndarray]:
    """
    Computes the top-k accuracy based on cosine similarity.

    Args:
    - k_similar_indices (np.ndarray): Indices of the top k most similar images.
    - true_label (int): The true label of the query image.
    - labels_memory_bank (np.ndarray): The labels of the images in the memory bank.
    - k (int): The number of nearest neighbors to check for accuracy.

    Returns:
    - Tuple[int, np.ndarray]: A tuple containing the accuracy (1 if true_label is in top k, else 0) and
      the list of predicted labels.
    """
    predictions = labels_memory_bank[k_similar_indices].squeeze()
    predictions = predictions[:k]
    correct = 0
    if true_label in predictions:
        correct += 1
    return 1 if correct > 0 else 0, predictions


def majority_vote_cosine_similarity(
        k_similar_indices: np.ndarray,
        true_label: int,
        labels_memory_bank: np.ndarray,
        k: int
) -> Tuple[int, np.ndarray]:
    """
    Performs majority voting to determine the most frequent label among the top k similar images.

    Args:
    - k_similar_indices (np.ndarray): Indices of the top k most similar images.
    - true_label (int): The true label of the query image.
    - labels_memory_bank (np.ndarray): The labels of the images in the memory bank.
    - k (int): The number of nearest neighbors to check for majority voting.

    Returns:
    - Tuple[int, np.ndarray]: A tuple containing the predicted label (1 for correct, 0 for incorrect)
      and the list of predicted labels.
    """
    predictions = labels_memory_bank[k_similar_indices]
    predictions = predictions[:k]
    true_label = int(true_label)

    counter = Counter(predictions.squeeze())
    frequent_labels = counter.most_common()
    max_frequency = frequent_labels[0][1]

    majority_labels = [label for label, freq in frequent_labels if freq == max_frequency]

    if len(majority_labels) == 1:
        mode_label = majority_labels[0]
    else:
        mode_label = min(majority_labels)

    prediction = 1 if mode_label == true_label else 0
    return prediction, predictions


def precision_cosine_similarity(
        k_similar_indices: np.ndarray,
        true_label: int,
        labels_memory_bank: np.ndarray,
        k: int
) -> float:
    """
    Computes precision for a query based on cosine similarity.

    Args:
    - k_similar_indices (np.ndarray): Indices of the top k most similar images.
    - true_label (int): The true label of the query image.
    - labels_memory_bank (np.ndarray): The labels of the images in the memory bank.
    - k (int): The number of nearest neighbors to check for precision.

    Returns:
    - float: The calculated precision value.
    """
    retrieved_labels = labels_memory_bank[k_similar_indices]
    retrieved_labels = retrieved_labels[:k]

    correct_retrievals = sum(1 for label in retrieved_labels if label == true_label)
    precision_value = correct_retrievals / k

    return precision_value


def create_confusion_matrix(
        data: pd.DataFrame,
        output_path: str,
        distance_metric: str
) -> None:
    """
    Creates and saves confusion matrices for different values of k based on the data provided.

    Args:
    - data (pd.DataFrame): A pandas DataFrame containing the following columns:
        - 'k' (int): The value of k for each sample, indicating how many nearest neighbors were considered.
        - 'label' (int): The true class label of the sample.
        - 'predictions' (list of int): A list of predicted class labels for each sample, generated by the model.
    - output_path (str): The directory path where the confusion matrix heatmap plots will be saved.
    - distance_metric (str): The distance metric used for generating predictions, included in the saved filename.

    Returns:
    - None: Saves confusion matrix plots to the specified output path.
    """
    num_classes = 4  # Number of classes in your classification problem
    class_names = [str(i) for i in range(num_classes)]  # Class labels as strings

    # Get unique k values from the dataset
    k_values = data['k'].unique()

    # Loop over each k value to compute and plot confusion matrices
    for k in k_values:
        # Filter the dataset for the current k value
        filtered_df = data.loc[data['k'] == k]

        # Initialize the confusion matrix as a square matrix of size num_classes x num_classes
        cm = np.zeros((num_classes, num_classes), dtype=int)

        # Loop over the rows of the filtered dataframe to build the confusion matrix
        for index, row in filtered_df.iterrows():
            real_label = row['label']  # True label
            predictions = row['predictions']  # Predicted labels (can be a list)

            # Update the confusion matrix: increment count for each predicted label
            for prediction in predictions:
                cm[real_label, prediction] += 1

        # Normalize the confusion matrix (divide by the sum of each row)
        row_sums = cm.sum(axis=1, keepdims=True)  # Sum of each row (true label)
        normalized_cm = cm / row_sums  # Element-wise division to normalize

        # Create DataFrames for both the regular and normalized confusion matrices
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        df_normalized_cm = pd.DataFrame(normalized_cm, index=class_names, columns=class_names)

        # Plotting settings
        sns.set(font_scale=1.5)
        font_size = 20

        # Plot and save regular confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": font_size})
        plt.xlabel("Predicted", fontsize=font_size)  # X-axis label
        plt.ylabel("Reference", fontsize=font_size)  # Y-axis label
        plt.title(f"Confusion Matrix for K={k} using {distance_metric}", fontsize=font_size)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, f'cm_{k}_{distance_metric}.png'))
        plt.clf()  # Clear the current figure for the next plot

        # Plot and save normalized confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_normalized_cm, annot=True, fmt=".2f", cmap="Blues", annot_kws={"size": font_size})
        plt.xlabel("Predicted", fontsize=font_size)  # X-axis label
        plt.ylabel("Reference", fontsize=font_size)  # Y-axis label
        plt.title(f"Normalized Confusion Matrix for K={k}", fontsize=font_size)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, f'cm_norm_{k}_{distance_metric}.png'))
        plt.clf()  # Clear the current figure for the next plot