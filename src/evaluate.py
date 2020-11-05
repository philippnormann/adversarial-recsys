import random
import argparse

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from pathlib import Path
from multiprocessing import cpu_count

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.helpers import cos_dist, create_index, get_nearest_neighbors_batch
from src.deepfashion.dataset import DeepFashionDataset, Split
from src.fashionnet.model import FashionNet, PredictionHead
from src.attacks.recommender import fgsm_sim_attack, pgd_sim_attack, cw_sim_attack
from src.attacks.classifier import fgsm_classifier_attack, pgd_classifier_attack
from src.attacks import AttackMethod


class EvalDataset(DeepFashionDataset):
    """Dataset for robustness evaluation of a k-NN recommender."""
    def __init__(self, num_samples, root_dir, split):
        """
        Args:
            num_samples (int): Number of article tuples to sample.
            root_dir (string): Path to directory with sampled annotations.
            split (Split): Split of dataset (test, train or both).
        """
        super().__init__(root_dir, split)
        self.attack_tuples = self.sample_attack_tuples(num_samples)

    def sample_attack_tuples(self, num_samples):
        random.seed(42)
        cat_labels = self.category_labels
        print(f'Sampling {num_samples} random attack tuples')
        attack_tuples = []
        t = tqdm(total=num_samples, leave=False)
        while len(attack_tuples) < num_samples:
            target_idx = random.randint(0, super().__len__() - 1)
            attack_idx = random.randint(0, super().__len__() - 1)
            target_cat = cat_labels.iloc[target_idx]['category_label']
            attack_cat = cat_labels.iloc[attack_idx]['category_label']
            # make sure random attack and target have different categories
            if target_cat != attack_cat:
                attack_tuples.append((target_idx, attack_idx))
                t.update()
        return attack_tuples

    def __len__(self):
        return len(self.attack_tuples)

    def __getitem__(self, idx):
        target_idx, attack_idx = self.attack_tuples[idx]
        target_example = super().__getitem__(target_idx)
        attack_exmaple = super().__getitem__(attack_idx)
        return {
            'target_name': target_example['name'],
            'target_image': target_example['image'],
            'attack_name': attack_exmaple['name'],
            'attack_image': attack_exmaple['image'],
        }


def evaluate_knn_recommender(net, index, embeddings, loader, epsilon, attack,
                             num_iterations):
    net.model.classifier = nn.Sequential()
    net.model.eval()

    print(f'Calculating {attack} adversarial ranks for eps={epsilon}')
    eps_results = pd.DataFrame(columns=[
        'target_name', 'attack_name', 'attack_rank', 'attack_dist',
        'adversarial_rank', 'adversarial_dist'
    ])
    for batch in tqdm(loader):
        target_names = batch['target_name']
        attack_names = batch['attack_name']
        attack_images = batch['attack_image'].to(device)

        target_vecs = embeddings.loc[target_names].values
        target_vecs = torch.tensor(target_vecs).detach().to(device)

        attack_vecs = embeddings.loc[attack_names].values
        attack_vecs = torch.tensor(attack_vecs).detach().to(device)

        if attack == AttackMethod.FGSM:
            adversarial_images = fgsm_sim_attack(net, target_vecs,
                                                 attack_images, epsilon)
        elif attack == AttackMethod.PGD:
            adversarial_images = pgd_sim_attack(net,
                                                target_vecs,
                                                attack_images,
                                                epsilon,
                                                k=num_iterations)
        elif attack == AttackMethod.CW:
            adversarial_images = cw_sim_attack(net,
                                               target_vecs,
                                               attack_images,
                                               epsilon=epsilon,
                                               n=num_iterations)
        else:
            parser.error(
                'unknown k-NN attack method, must be one of: fgsm, pgd, cw')

        adversarial_vecs = net.model(adversarial_images).detach()
        adversarial_dists = cos_dist(target_vecs,
                                     adversarial_vecs).cpu().numpy()

        attack_dists = cos_dist(target_vecs, attack_vecs).cpu().numpy()
        attack_rank_ids = [
            embeddings.index.get_loc(name) for name in attack_names
        ]
        attack_rank_ids = np.expand_dims(attack_rank_ids, 1)

        target_nn_ids, target_nn_dists = get_nearest_neighbors_batch(
            index, target_vecs.cpu(), len(embeddings))
        attack_ranks = np.where(target_nn_ids == attack_rank_ids)[1]
        adversarial_ranks = np.argmax(
            np.expand_dims(adversarial_dists, axis=1) <= target_nn_dists,
            axis=1)

        batch_results = {
            'target_name': target_names,
            'attack_name': attack_names,
            'attack_rank': attack_ranks,
            'attack_dist': attack_dists,
            'adversarial_rank': adversarial_ranks,
            'adversarial_dist': adversarial_dists
        }
        eps_results = eps_results.append(pd.DataFrame(batch_results),
                                         ignore_index=True)
    return eps_results


def calc_success_rates(ranks,
                       min_ranks=[1, 3, 5, 10, 20, 30, 40, 50, 100, 500,
                                  1000]):
    success_rates = pd.DataFrame(columns=['rank', 'success_rate'])
    for min_rank in min_ranks:
        success_rate = 100 * (ranks.adversarial_rank <=
                              min_rank - 1).sum() / len(ranks)
        success_rates = success_rates.append(
            {
                'rank': min_rank,
                'success_rate': success_rate
            },
            ignore_index=True)
    return success_rates


def evaluate_classifier(net, loader, attack=True):
    running_cat_loss = 0.0
    running_cat_corrects = 0.0
    running_cat_top5_corrects = 0.0

    running_tex_loss = 0.0
    running_tex_top1_corrects = 0.0

    category_criterion = nn.CrossEntropyLoss()
    texture_criterion = nn.BCEWithLogitsLoss()

    print('Testing model...')
    for batch in tqdm(loader):
        category_labels = batch['category'].to(device)
        texture_labels = batch['textures'].to(device)
        inputs = batch['image'].to(device)

        if attack:
            inputs = pgd_classifier_attack(net, inputs, category_labels,
                                           texture_labels, category_criterion,
                                           texture_criterion)

        category_outputs, texture_outputs = net.model(inputs)

        category_loss = category_criterion(category_outputs, category_labels)
        texture_loss = texture_criterion(texture_outputs, texture_labels)

        category_preds = torch.argmax(category_outputs, dim=1)
        _, top5_category_preds = category_outputs.topk(5, dim=1)

        texture_preds = texture_outputs.argmax(dim=1)
        example_indices = torch.arange(0, texture_outputs.size(0)).to(device)
        texture_preds_with_idx = torch.stack([example_indices, texture_preds],
                                             dim=1).unsqueeze(1)
        texture_top1_corrects = torch.sum(
            torch.sum(texture_labels.nonzero() == texture_preds_with_idx,
                      dim=2) == 2)

        running_cat_loss += category_loss.item() * inputs.size(0)
        running_cat_corrects += torch.sum(category_preds == category_labels)
        running_cat_top5_corrects += torch.sum(
            top5_category_preds == category_labels.unsqueeze(1))

        running_tex_loss += texture_loss.item() * inputs.size(0)
        running_tex_top1_corrects += texture_top1_corrects

    cat_loss = running_cat_loss / len(loader.dataset)
    cat_acc = running_cat_corrects.double() / len(loader.dataset)
    cat_top5_acc = running_cat_top5_corrects.double() / len(loader.dataset)

    tex_loss = running_tex_loss / len(loader.dataset)
    tex_top1_prec = running_tex_top1_corrects.double() / len(loader.dataset)

    print(f'Category Loss = {cat_loss}')
    print(f'Category Accuracy = {cat_acc.item()*100}%')
    print(f'Category Top-5 Accuracy = {cat_top5_acc.item()*100}%')

    print(f'Texture Loss = {tex_loss}')
    print(f'Texture Top-1 Precision = {tex_top1_prec.item()*100}%')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate quality and robustness of model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-name',
                        help="saved model to attack",
                        type=str,
                        default='normal-24-epochs')

    parser.add_argument('--split',
                        help="data split to use for evaluation",
                        type=Split,
                        choices=list(Split),
                        default=Split.TEST)

    parser.add_argument('--output-dir',
                        help="output directory for evaluation results",
                        type=str,
                        default='results/evaluation')

    parser.add_argument('--batch-size',
                        help="batch size to use for attacks",
                        type=int,
                        default=32)

    parser_eval_type = parser.add_subparsers(dest='eval_type',
                                             title='evaluation types',
                                             required=True)

    parser_knn = parser_eval_type.add_parser('knn')
    parser_knn.add_argument(
        '--num-samples',
        help="number of article tuples to use for evaluation",
        type=int,
        default=1000)
    parser_knn.add_argument('--attack',
                            help="attack method to use for evaluation",
                            type=AttackMethod,
                            choices=list(AttackMethod),
                            default=None)

    parser_knn.add_argument('--num-iterations',
                            help="iterations for iterative attacks",
                            type=int,
                            default=32)

    parser_knn.add_argument('--epsilons',
                            nargs='+',
                            default=[0.01, 0.02, 0.03, 0.04, 0.05],
                            type=float)

    parser_classifier = parser_eval_type.add_parser('classifier')
    parser_classifier.add_argument(
        '--attack',
        help="attack classifier using PGD during evaluation",
        action='store_true')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ' + str(device))

    out_dir = f'{args.output_dir}/{args.model_name}/{args.split}'

    if args.eval_type == 'knn':
        print(f'Creating {args.split} dataset n={args.num_samples}')
        eval_dataset = EvalDataset(num_samples=args.num_samples,
                                   root_dir='./data/DeepFashion/',
                                   split=args.split)
        data_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=cpu_count())

        print(f'Loading model: {args.model_name}')
        fashion_net = FashionNet(device, eval_dataset.num_categories,
                                 eval_dataset.num_textures)
        fashion_net.load(f'results/models/{args.model_name}.pt')

        print('Creating NMSLIB index for efficient k-NN search')
        index, embeddings = create_index('./data/DeepFashion/',
                                         args.model_name)

        rank_results = {}
        success_results = {}

        if args.attack == AttackMethod.FGSM:
            out_file_prefix = f'{args.attack}'
        else:
            out_file_prefix = f'{args.attack}-{args.num_iterations}'

        print(f'Starting k-NN robustness evaluation for Ɛ={args.epsilons}')
        for eps in args.epsilons:
            try:
                rank_results[eps] = pd.read_csv(
                    f'{out_dir}/{eps}/{out_file_prefix}-knn.csv')
                rank_results[eps].drop('Unnamed: 0',
                                       axis=1,
                                       inplace=True,
                                       errors='ignore')
            except FileNotFoundError:
                rank_results[eps] = evaluate_knn_recommender(
                    fashion_net, index, embeddings, data_loader, eps,
                    args.attack, args.num_iterations)
            try:
                success_results[eps] = pd.read_csv(
                    f'{out_dir}/{eps}/{out_file_prefix}-knn-success.csv')
            except FileNotFoundError:
                print(f'Calculating success rates for eps={eps}')
                success_results[eps] = calc_success_rates(rank_results[eps])
                print(success_results[eps])

        for eps in args.epsilons:
            print(f'Writing results to {out_dir}/{eps}/{out_file_prefix}')
            Path(f'{out_dir}/{eps}').mkdir(exist_ok=True, parents=True)
            rank_results[eps].to_csv(
                f'{out_dir}/{eps}/{out_file_prefix}-knn.csv', index=False)
            success_results[eps].to_csv(
                f'{out_dir}/{eps}/{out_file_prefix}-knn-success.csv',
                index=False)

    elif args.eval_type == 'classifier':
        print(f'Creating {args.split} dataset')
        eval_dataset = DeepFashionDataset(root_dir='./data/DeepFashion/',
                                          split=args.split)
        data_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=cpu_count())

        print(f'Loading model: {args.model_name}')
        fashion_net = FashionNet(device, eval_dataset.num_categories,
                                 eval_dataset.num_textures)
        fashion_net.load(f'results/models/{args.model_name}.pt')
        fashion_net.model.eval()

        evaluate_classifier(fashion_net, data_loader, args.attack)

    else:
        parser.error('unknown eval type, must be one of: knn, classifier')
