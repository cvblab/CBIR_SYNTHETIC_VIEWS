import argparse
import os
import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
from model import Model
import random
import configparser
import ast
import ssl
from evaluation import evaluation

ssl._create_default_https_context = ssl._create_unverified_context


def train(net: torch.nn.Module, data_loader: DataLoader, train_optimizer: optim.Optimizer,
          progleason_model: torch.nn.Module, device: str, temperature: float,
          batch_size: int, noise_dim: int, epoch: int, epochs: int) -> float:
    """
    Train the model for one epoch.

    Args:
        net (torch.nn.Module): The model to be trained.
        data_loader (DataLoader): DataLoader with training data.
        train_optimizer (optim.Optimizer): Optimizer for updating model parameters.
        progleason_model (torch.nn.Module): Synthetic data generator model.
        device (str): Device ('cuda' or 'cpu').
        temperature (float): Temperature for similarity calculation.
        batch_size (int): Training batch size.
        noise_dim (int): Noise dimension for synthetic data generation.
        epoch (int): Current training epoch.
        epochs (int): Total number of training epochs.

    Returns:
        float: Average loss for this epoch.
    """
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, _, target, path in train_bar:
        pos_1 = pos_1.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        pos_2 = utils.create_synthetic_similar_sample(pos_1.shape[0], target, progleason_model, noise_dim, device)

        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        out = torch.cat([out_1, out_2], dim=0)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description(f'Train Epoch: [{epoch}/{epochs}] Loss: {total_loss / total_num:.4f}')

    return total_loss / total_num


def test(net: torch.nn.Module, memory_data_loader: DataLoader, test_data_loader: DataLoader,
         k: int, temperature: float, c: int, epoch: int, epochs: int) -> tuple:
    """
    Evaluate the model using k-NN weighted search.

    Args:
        net (torch.nn.Module): The model being evaluated.
        memory_data_loader (DataLoader): DataLoader for memory (stored features).
        test_data_loader (DataLoader): DataLoader for test data.
        k (int): Number of neighbors for k-NN search.
        temperature (float): Temperature for similarity calculation.
        c (int): Number of classes.
        epoch (int): Current test epoch.
        epochs (int): Total number of test epochs.

    Returns:
        tuple: Top-1 and top-5 accuracy for this epoch.
    """
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    feature_labels = torch.tensor([]).cuda(non_blocking=True)
    with torch.no_grad():
        # Generate feature bank
        for data, _, target, _ in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
            target = target.cuda(non_blocking=True)
            feature_labels = torch.cat([feature_labels, target.unsqueeze(1)], dim=0)

        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()

        # Evaluate on test set
        test_bar = tqdm(test_data_loader)
        feature_labels = feature_labels.squeeze()
        for data, _, target, _ in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            sim_matrix = torch.mm(feature, feature_bank)
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).to(torch.int64), value=1.0)
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description(
                f'Test Epoch: [{epoch}/{epochs}] Acc@1:{total_top1 / total_num * 100:.2f}% Acc@5:{total_top5 / total_num * 100:.2f}%')

    return total_top1 / total_num * 100, total_top5 / total_num * 100


def main(path_experiment: str, last_epoch_value: int = 0) -> None:
    """
    Main function that manages data loading, training, and model evaluation.

    Args:
        path_experiment (str): Path where configuration files and results are located.
        last_epoch_value (int, optional): The value of the last trained epoch, default is 0.
    """
    # Load parameters from the config.ini file
    config_path = os.path.join(path_experiment, 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_path)

    path_csv_sicap_test = config['VARIABLES']['path_csv_sicap_test']
    path_sicap_images = config['VARIABLES']['path_sicap_images']
    path_sicap_csv = config['VARIABLES']['path_sicap_csv']
    channels_img = int(config['VARIABLES']['channels_img'])
    num_workers = int(config['VARIABLES']['num_workers'])
    results_path = config['VARIABLES']['results_path']
    evaluation_path = config['VARIABLES']['evaluation_path']
    progleason_model = config['VARIABLES']['progleason_model']
    z_dim = int(config['VARIABLES']['z_dim'])
    noise_dim = int(config['VARIABLES']['noise_dim'])
    temperature = float(config['VARIABLES']['temperature'])
    k = int(config['VARIABLES']['k'])
    batch_size = int(config['VARIABLES']['batch_size'])
    epochs = int(config['VARIABLES']['epochs'])
    subset = ast.literal_eval(config['VARIABLES']['subset'])
    seed = int(config['VARIABLES']['seed'])
    lr = float(config['VARIABLES']['lr'])

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Data preparation
    train_data = utils.SiCAPv2(csv_file=path_sicap_csv, root_dir=path_sicap_images, subset=subset)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                              drop_last=True)

    memory_data = utils.SiCAPv2(csv_file=path_sicap_csv, root_dir=path_sicap_images, transform=utils.test_transform,
                                subset=subset)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                               pin_memory=True)

    test_data = utils.SiCAPv2(csv_file=path_csv_sicap_test, root_dir=path_sicap_images, transform=utils.test_transform,
                              subset=subset)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model and optimizer configuration
    model = Model(z_dim).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print(f'# Model Params: {params} FLOPs: {flops}')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    c = 4

    # Training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = f'{z_dim}_{temperature}_{k}_{batch_size}_{epochs}'
    best_acc = 0.0

    if last_epoch_value != 0:
        df = pd.read_csv(os.path.join(results_path, f'{save_name_pre}_statistics.csv'))
        for key in results.keys():
            if key in df.columns:
                results[key] = df[key].tolist()
                path_model = os.path.join(results_path, f'{save_name_pre}_model.pth')
        model = utils.load_pretrained_encoder(path_model, z_dim)
        model.train()

    # Load progleason model
    progleason_model = utils.load_synthetic_model(progleason_model, device)

    # Training loop
    for epoch in range(last_epoch_value + 1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, progleason_model, device, temperature, batch_size, noise_dim,
                           epoch, epochs)
        results['train_loss'].append(train_loss)

        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, k, temperature, c, epoch, epochs)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)

        # Save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(os.path.join(results_path, f'{save_name_pre}_statistics.csv'), index_label='epoch')
        utils.plot_curves(data_frame, results_path)

        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), os.path.join(results_path, f'{save_name_pre}_model.pth'))

    evaluation(evaluation_path, train_loader, test_loader, os.path.join(results_path, f'{save_name_pre}_model.pth'),
               z_dim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Training and Evaluation")
    parser.add_argument('--path_experiment', type=str, required=True,
                        help="Path of the experiment where configuration files and results are located.")
    parser.add_argument('--last_epoch', type=int, default=0, help="Number of the last trained epoch.")
    args = parser.parse_args()

    main(args.path_experiment, args.last_epoch)

