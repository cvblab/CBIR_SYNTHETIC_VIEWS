import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import utils
import time


def evaluation(output_path_results: str, train_loader: torch.utils.data.DataLoader,
               test_loader: torch.utils.data.DataLoader,
               path_model: str, z_dim: int) -> None:
    """
    Evaluates a model using cosine similarity and various metrics.

    Parameters:
    output_path_results (str): Path to save evaluation results.
    train_loader (torch.utils.data.DataLoader): DataLoader for training dataset.
    test_loader (torch.utils.data.DataLoader): DataLoader for test dataset.
    path_model (str): Path to the pre-trained model.
    z_dim (int): Dimension of the encoded feature space.

    Returns:
    None: Saves evaluation results to files.
    """

    if not os.path.exists(output_path_results):
        os.makedirs(output_path_results)

    loop = tqdm(test_loader, leave=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the encoder model
    encoder = utils.load_pretrained_encoder(path_model, z_dim)
    create_database = True
    save_retrievals = False
    compute_ssim = False
    compute_tsne = False

    # Define output directories
    output_path = os.path.join(output_path_results, 'memory_bank_test_sicap')
    output_path_total = os.path.join(output_path_results, 'memory_bank_test_sicap_total')
    output_retrievals = os.path.join(output_path_results, 'retrievals')
    output_path_confusion_matrix_results = os.path.join(output_path_results, 'confusion_matrices')
    output_path_tsne = os.path.join(output_path_results, 'representations')

    for path in [output_path, output_path_total, output_retrievals, output_path_confusion_matrix_results,
                 output_path_tsne]:
        if not os.path.exists(path):
            os.makedirs(path)

    if create_database:
        utils.create_database(output_path, output_path_total, encoder, train_loader, device)

    # Load memory bank data
    memory_bank = np.load(os.path.join(output_path_total, 'memory_bank_total.npy'))
    labels_memory_bank = np.load(os.path.join(output_path_total, 'memory_bank_labels_total.npy'))
    paths_memory_bank = np.load(os.path.join(output_path_total, 'memory_bank_paths_total.npy'))

    k_values = [1, 3, 5, 7]
    cosine_similarity_times = []
    encoded_patches_list = []
    labels_patches = []

    with torch.no_grad():
        for batch_idx, data in enumerate(loop):
            imgs, _, labels, paths = data
            imgs = imgs.to(device)
            _, encoded_patches = encoder(imgs)

            for i, encoded_patch in enumerate(encoded_patches):
                if compute_tsne:
                    encoded_patches_list.append(encoded_patch.cpu().detach().numpy())
                    labels_patches.append(labels[i].item())

                start_time = time.time()
                k_similar_indices = utils.compute_cosine_similarity(encoded_patch, memory_bank, 7)
                cosine_similarity_times.append(time.time() - start_time)

    cosine_similarity_mean_time = sum(cosine_similarity_times) / len(cosine_similarity_times)

    # Save computation time to an Excel file
    df_time = pd.DataFrame({'Cosine Similarity Mean Time': [cosine_similarity_mean_time]})
    df_time.to_excel(os.path.join(output_path_results, 'times.xlsx'))

    # Generate t-SNE representations if required
    if compute_tsne:
        utils.tsne_representations(encoded_patches_list, labels_patches, output_path_tsne)

    # Create confusion matrix
    utils.create_confusion_matrix([], output_path_confusion_matrix_results, 'cosine')

