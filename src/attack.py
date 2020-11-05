import argparse

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from src.helpers import cos_dist
from src.deepfashion.dataset import DeepFashionDataset, INVERSE_IMG_TRANSFORM
from src.fashionnet.model import FashionNet, PredictionHead
from src.attacks.recommender import fgsm_sim_attack, pgd_sim_attack, cw_sim_attack

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Perform a single attack using FGSM, PGD or CW',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-name',
                        help="saved model to attack",
                        type=str,
                        default='normal-24-epochs')

    parser.add_argument(
        '--attack-article',
        help="original attack article image",
        type=str,
        default='img_resized/Linen-Blend_Drawstring_Shorts/img_00000026.jpg')

    parser.add_argument(
        '--target-article',
        help="target article image",
        type=str,
        default='img_resized/Striped_Textured_Sweater/img_00000024.jpg')

    parser.add_argument('--output-dir',
                        help="output directory for result images",
                        type=str,
                        default='results/attack')

    parser.add_argument('--epsilon',
                        help="maximum pertubation",
                        type=float,
                        default=0.03)

    parser.add_argument('--no-plot',
                        help="disable interactive plots",
                        action='store_true')

    subparsers = parser.add_subparsers(dest='attack_method',
                                       title='attack methods',
                                       required=True)

    parser_fgsm = subparsers.add_parser('fgsm')

    parser_pgd = subparsers.add_parser('pgd')
    parser_pgd.add_argument('--num-iterations', type=int, default=32)

    parser_cw = subparsers.add_parser('cw')
    parser_cw.add_argument('--num-iterations', type=int, default=1000)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ' + str(device))

    out_dir = f'{args.output_dir}/{args.model_name}/{args.attack_method}/{args.epsilon}'
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    print('Loading DeepFashion dataset')
    fashion_dataset = DeepFashionDataset(root_dir='./data/DeepFashion/')
    fashion_net = FashionNet(device, fashion_dataset.num_categories,
                             fashion_dataset.num_textures)

    print(f'Loading model: {args.model_name}')
    fashion_net.load(f'results/models/{args.model_name}.pt')
    fashion_net.model.classifier = nn.Sequential()
    fashion_net.model.eval()

    target_id = fashion_dataset.category_labels.index.get_loc(
        args.target_article)
    target_image = fashion_dataset[target_id]['image'].to(device)
    target_vec = fashion_net.model(target_image.unsqueeze(dim=0)).detach()
    target_image_pillow = INVERSE_IMG_TRANSFORM(target_image.cpu())
    target_image_pillow.save(f'{args.output_dir}/target.jpg')

    attack_id = fashion_dataset.category_labels.index.get_loc(
        args.attack_article)
    attack_image = fashion_dataset[attack_id]['image'].to(device)
    attack_vec = fashion_net.model(attack_image.unsqueeze(dim=0)).detach()
    attack_image_pillow = INVERSE_IMG_TRANSFORM(attack_image.cpu())
    attack_image_pillow.save(f'{args.output_dir}/original.jpg')

    print(
        f'Performing attack using, {args.attack_method} method, eps={args.epsilon}'
    )
    if args.attack_method == 'fgsm':
        adversarial_image = fgsm_sim_attack(fashion_net,
                                            target_vec,
                                            attack_image.unsqueeze(0),
                                            epsilon=args.epsilon)[0]
    elif args.attack_method == 'pgd':
        adversarial_image = pgd_sim_attack(fashion_net,
                                           target_vec,
                                           attack_image.unsqueeze(0),
                                           epsilon=args.epsilon,
                                           k=args.num_iterations)[0]
    elif args.attack_method == 'cw':
        adversarial_image = cw_sim_attack(fashion_net,
                                          target_vec,
                                          attack_image.unsqueeze(0),
                                          epsilon=args.epsilon,
                                          n=args.num_iterations)[0]
    else:
        parser.error('unknown attack method, must be one of: fgsm, pgd, cw')

    adversarial_vec = fashion_net.model(adversarial_image.unsqueeze(dim=0))

    dist_before = cos_dist(target_vec, attack_vec)[0].item()
    dist_after = cos_dist(target_vec, adversarial_vec)[0].item()

    print(f'Cosine distance before attack: {dist_before}')
    print(f'Cosine distance after attack: {dist_after}')

    dist_df = pd.DataFrame(data=[(dist_before, dist_after)],
                           columns=['dist_before', 'dist_after'])
    dist_df.to_csv(f'{out_dir}/distances.csv', index=False)

    final_pertubation = adversarial_image - attack_image
    print(
        f'L_inf norm after attack: {torch.max(torch.abs(final_pertubation)):.2f}'
    )
    min_pertubation = torch.min(final_pertubation)
    max_pertubation = torch.max(final_pertubation)
    visualized_pertubation = (final_pertubation - min_pertubation) / (
        max_pertubation - min_pertubation)

    INVERSE_IMG_TRANSFORM(
        visualized_pertubation.cpu()).save(f'{out_dir}/pertubation.jpg')
    INVERSE_IMG_TRANSFORM(
        adversarial_image.cpu()).save(f'{out_dir}/attack.jpg')

    with open(f'{out_dir}/adversarial_examples.tsv', 'wt') as f:
        f.write('image\t' +
                '\t'.join(map(lambda x: 'd' + str(x), range(1280))) + '\n')
        f.write(f'../../{out_dir}/attack.jpg\t' + '\t'.join(
            map(str, list(adversarial_vec.cpu().detach().numpy()[0]))) + '\n')

    if not args.no_plot:
        fig1 = plt.figure()
        fig1.canvas.set_window_title('Target Article')
        plt.imshow(target_image_pillow)

        fig2 = plt.figure()
        fig2.canvas.set_window_title('Original Attack Article')
        plt.imshow(attack_image_pillow)

        fig3 = plt.figure()
        fig3.canvas.set_window_title(
            f'{args.attack_method.upper()} pertubation (epsilon={args.epsilon})'
        )
        plt.imshow(INVERSE_IMG_TRANSFORM(visualized_pertubation.cpu()))

        fig4 = plt.figure()
        fig4.canvas.set_window_title(
            f'Adversarial {args.attack_method.upper()} example')
        plt.imshow(INVERSE_IMG_TRANSFORM(adversarial_image.cpu()))

        plt.show()
