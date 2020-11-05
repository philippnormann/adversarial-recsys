import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from enum import Enum


class Split(Enum):
    TEST = 'test'
    TRAIN = 'train'
    BOTH = 'both'

    def __str__(self):
        return self.value


DEFAULT_IMG_TRANSFORM = transforms.Compose(
    [transforms.Resize(224), transforms.ToTensor()])

INVERSE_IMG_TRANSFORM = transforms.Compose([transforms.ToPILImage()])


class DeepFashionDataset(Dataset):
    """Deep Fashion dataset."""
    def __init__(self,
                 root_dir,
                 split=Split.BOTH,
                 img_transforms=DEFAULT_IMG_TRANSFORM):
        """
        Args:
            root_dir (string): Path to directory with processed annotations.
            img_transforms (callable, optional): Transform to be applied.
        """
        self.root_dir = root_dir
        self.split = split
        self.partitions = pd.read_csv(root_dir +
                                      'eval_processed/eval_partition.tsv',
                                      sep='\t')
        self.partitions = self.partitions.replace('val', 'train')

        self.category_names = pd.read_csv(root_dir +
                                          'anno_processed/category_names.tsv',
                                          sep='\t')

        self.texture_names = pd.read_csv(root_dir +
                                         'anno_processed/texture_names.tsv',
                                         sep='\t')

        self.num_categories = len(self.category_names)
        self.num_textures = len(self.texture_names)

        self.category_labels = pd.read_csv(
            root_dir + 'anno_processed/category_labels.tsv', sep='\t')
        self.texture_labels = pd.read_json(
            root_dir + 'anno_processed/texture_labels.jsonl', lines=True)

        self.partitions.image_name = self.partitions.image_name.str.replace(
            'img/', 'img_resized/')
        self.category_labels.image_name = self.category_labels.image_name.str.replace(
            'img/', 'img_resized/')
        self.texture_labels.image_name = self.texture_labels.image_name.str.replace(
            'img/', 'img_resized/')

        self.partitions = self.partitions.set_index('image_name')
        self.category_labels = self.category_labels.set_index('image_name')
        self.texture_labels = self.texture_labels.set_index('image_name')

        if self.split != Split.BOTH:
            self.category_labels = self.category_labels.join(self.partitions)
            self.category_labels = self.category_labels[
                self.category_labels.evaluation_status == self.split.value]

            self.texture_labels = self.texture_labels.join(self.partitions)
            self.texture_labels = self.texture_labels[
                self.texture_labels.evaluation_status == self.split.value]

        assert (len(self.category_labels) == len(self.texture_labels))

        self.img_transforms = img_transforms

    def __len__(self):
        return len(self.category_labels)

    def __getitem__(self, idx):
        category_row = self.category_labels.iloc[idx]

        image_name = category_row.name
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.img_transforms(image)

        category = category_row.category_label

        textures = self.texture_labels.loc[image_name].textures
        texture_vector = torch.zeros(self.num_textures)
        for textures_id in textures:
            texture_vector[textures_id] = 1.0

        return {
            'image': image,
            'name': image_name,
            'category': category,
            'textures': texture_vector
        }

    def plot_image_batch(self, batch, images_per_row=8):
        batch_size = len(batch['name'])
        num_rows = batch_size // images_per_row
        num_cols = images_per_row

        fig, axes = plt.subplots(num_rows, num_cols)
        fig.set_figheight(num_rows * 5)
        fig.set_figwidth(num_cols * 3)

        for idx in range(batch_size):
            row_idx = idx // num_cols
            col_idx = idx % num_cols
            cat_name = self.category_names.iloc[int(
                batch['category'][idx])].category_name
            tex_ids = np.argwhere(batch['textures'][idx] != 0).reshape(-1)
            tex_names = '\n'.join([
                self.texture_names.iloc[int(i)].attribute_name for i in tex_ids
            ])
            image = INVERSE_IMG_TRANSFORM(batch['image'][idx])
            axes[row_idx, col_idx].imshow(image)
            axes[row_idx,
                 col_idx].set_title('\huge{' + cat_name + '}\n' + tex_names)

        plt.show()

    def visualize_predictions(self, fashionnet, batch):
        def top_10_predictions(probabilities, names):
            score_name_tuples = zip(names, probabilities)
            sorted_tuples = sorted(score_name_tuples, key=lambda x: x[1])
            top_10_tuples = list(sorted_tuples)[-10:]
            names = [x[0] for x in top_10_tuples]
            probs = [x[1] for x in top_10_tuples]
            return names, probs

        def color_predictions(predicted_names, correct_names):
            wrong_color = '#2d98da'
            correct_color = '#20bf6b'
            return [
                correct_color if name in correct_names else wrong_color
                for name in predicted_names
            ]

        batch_size = len(batch['name'])
        category_logits, texture_logits = fashionnet.model(batch['image'].to(
            fashionnet.device))
        category_logits = category_logits.detach().cpu()
        texture_logits = texture_logits.detach().cpu()

        category_softmax = torch.exp(
            torch.nn.functional.log_softmax(category_logits, dim=1)).detach()
        predicted_category_id = torch.argmax(category_softmax, dim=1)

        texture_softmax = torch.exp(
            torch.nn.functional.log_softmax(texture_logits, dim=1)).detach()
        predicted_texture_id = torch.argmax(texture_softmax, dim=1)

        for idx in range(batch_size):
            fig, axes = plt.subplots(1, 3)
            fig.set_figheight(6)
            fig.set_figwidth(18)

            category_label_name = self.category_names.iloc[int(
                batch['category'][idx])].category_name
            texture_label_ids = np.argwhere(
                batch['textures'][idx] != 0).reshape(-1)
            texture_label_names = [
                self.texture_names.iloc[int(i)].attribute_name
                for i in texture_label_ids
            ]

            top_10_cat_names, top_10_cat_probs = top_10_predictions(
                category_softmax[idx],
                self.category_names.category_name.values)
            top_10_tex_names, top_10_tex_probs = top_10_predictions(
                texture_softmax[idx], self.texture_names.attribute_name.values)

            image = INVERSE_IMG_TRANSFORM(batch['image'][idx])

            axes[0].set_title('\huge{' + category_label_name + '}\n' +
                              '\n'.join(texture_label_names))
            axes[0].imshow(image)

            predicted_category_name = self.category_names.iloc[int(
                predicted_category_id[idx])].category_name
            axes[1].set_title('\huge{Category: ' + predicted_category_name +
                              '}')
            axes[1].barh(top_10_cat_names,
                         width=top_10_cat_probs,
                         color=color_predictions(top_10_cat_names,
                                                 [category_label_name]))
            print(top_10_cat_probs)
            print(top_10_cat_names)

            predicted_texture_name = self.texture_names.iloc[int(
                predicted_texture_id[idx])].attribute_name
            axes[2].set_title('\huge{Texture: ' + predicted_texture_name + '}')
            axes[2].barh(top_10_tex_names,
                         width=top_10_tex_probs,
                         color=color_predictions(top_10_tex_names,
                                                 texture_label_names))
            print(top_10_tex_probs)
            print(top_10_tex_names)

            plt.show()
