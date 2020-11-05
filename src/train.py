import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from multiprocessing import cpu_count
from pathlib import Path

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.deepfashion.dataset import DeepFashionDataset, Split
from src.fashionnet.model import FashionNet, PredictionHead
from src.defenses.training import adversarial_train, curriculum_adversarial_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train FashionNet classification model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch-size', type=int, default=32)

    subparsers = parser.add_subparsers(dest='train_method',
                                       title='train methods',
                                       required=True)

    parser_normal = subparsers.add_parser('normal')
    parser_normal.add_argument('--num-epochs', type=int, default=24)

    parser_adv = subparsers.add_parser('adversarial')
    parser_adv.add_argument('--num-epochs', type=int, default=24)
    parser_adv.add_argument('--attack-strength', type=int, default=8)

    parser_curriculum_adv = subparsers.add_parser('curriculum-adversarial')
    parser_curriculum_adv.add_argument('--max-attack-strength',
                                       type=int,
                                       default=8)
    parser_curriculum_adv.add_argument('--epoch-overfit-threshold',
                                       type=int,
                                       default=10)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ' + str(device))

    preprocessing = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.25), ratio=(1.0, 1.0)),
        transforms.RandomRotation(15, fill=255),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    train_set = DeepFashionDataset(root_dir='./data/DeepFashion/',
                                   split=Split.TRAIN,
                                   img_transforms=preprocessing)

    test_set = DeepFashionDataset(root_dir='./data/DeepFashion/',
                                  split=Split.TEST,
                                  img_transforms=preprocessing)

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=cpu_count())

    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             num_workers=cpu_count())

    data_loaders = {'train': train_loader, 'test': test_loader}

    fashion_net = FashionNet(device, train_set.num_categories,
                             train_set.num_textures)

    category_criterion = nn.CrossEntropyLoss()
    texture_criterion = nn.BCEWithLogitsLoss()

    parameters = fashion_net.model.parameters()
    optimizer = optim.Adam(parameters, lr=0.001)

    if args.train_method == 'normal':
        model_name = f'normal-{args.num_epochs}-epochs'
        writer = SummaryWriter(log_dir='results/runs/' + model_name)
        fashion_net.train(data_loaders,
                          category_criterion,
                          texture_criterion,
                          optimizer,
                          writer,
                          num_epochs=args.num_epochs)

    elif args.train_method == 'adversarial':
        model_name = f'adversarial-{args.num_epochs}-epochs'
        writer = SummaryWriter(log_dir='results/runs/' + model_name)
        adversarial_train(fashion_net,
                          data_loaders,
                          category_criterion,
                          texture_criterion,
                          optimizer,
                          writer,
                          num_epochs=args.num_epochs,
                          attack_strength=args.attack_strength)

    elif args.train_method == 'curriculum-adversarial':
        model_name = f'curriculum-adversarial-{args.max_attack_strength}-k'
        writer = SummaryWriter(log_dir='results/runs/' + model_name)
        curriculum_adversarial_train(
            fashion_net,
            data_loaders,
            category_criterion,
            texture_criterion,
            optimizer,
            writer,
            max_k=args.max_attack_strength,
            epoch_overfit_threshold=args.epoch_overfit_threshold)
    else:
        parser.error(
            'unknown train method, must be one of: normal, adversarial, curriculum-adversarial'
        )

    Path('results/models/').mkdir(exist_ok=True, parents=True)
    fashion_net.save('results/models/' + model_name + '.pt')
