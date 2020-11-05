import json
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm
from imagededup.methods import PHash

data_dir = 'data/DeepFashion/'

category_names_file = data_dir + 'anno/list_category_cloth.txt'
category_labels_file = data_dir + 'anno/list_category_img.txt'

attribute_names_file = data_dir + 'anno/list_attr_cloth.txt'
attribute_labels_file = data_dir + 'anno/list_attr_img.txt'

eval_partitions_file = data_dir + 'eval/list_eval_partition.txt'


def remove_duplicates(data_dir, image_names):
    phasher = PHash()
    hashed_images = dict()
    unique_images = set()
    for img in tqdm(image_names):
        img_hash = phasher.encode_image(data_dir + img)
        hashed_images.setdefault(img_hash, []).append(img)
    for phash, imgs in hashed_images.items():
        unique_images.add(imgs[0])
    return unique_images


def read_tuple_format(txt_path):
    with open(txt_path, 'rt') as f:
        num_entries = int(f.readline())
        column_1, column_2 = f.readline().rstrip().split()
        cat_splits = [line.rstrip().split() for line in f]
        data_dict = {
            column_1: [' '.join(split[:-1]).rstrip() for split in cat_splits],
            column_2: [split[-1] for split in cat_splits]
        }
        assert (len(cat_splits) == num_entries)
        df = pd.DataFrame.from_dict(data_dict)
        return df


def read_vector_format(txt_path):
    data_dict = {}
    with open(txt_path, 'rt') as f:
        num_entries = int(f.readline())
        index_name, vector_column = f.readline().rstrip().split()
        for line in tqdm(f, total=num_entries):
            line_split = line.rstrip().split()
            dense_vec = np.array([int(x) for x in line_split[1:]])
            sparse_indices = np.argwhere(dense_vec != -1).flatten().tolist()
            data_dict[line_split[0]] = sparse_indices
        assert (len(data_dict) == num_entries)
        return data_dict


def save_vector_dict(vec_dict, path):
    with open(path, 'wt') as f:
        for image_name, attributes in tqdm(vec_dict.items()):
            row = {'image_name': image_name, 'textures': attributes}
            f.write(json.dumps(row) + '\n')


if __name__ == "__main__":
    print('Reading category names...')
    category_names = read_tuple_format(category_names_file)
    category_names.category_type = category_names.category_type.apply(int)
    print(category_names.head())

    print('Reading category labels...')
    category_labels = read_tuple_format(category_labels_file)
    category_labels = category_labels.set_index('image_name')
    category_labels.category_label = category_labels.category_label.apply(int)
    category_labels.category_label -= 1
    print(category_labels.head())

    print('Reading attribute names...')
    attribute_names = read_tuple_format(attribute_names_file)
    attribute_names.attribute_type = attribute_names.attribute_type.apply(int)
    print(attribute_names.head())

    print('Reading attribute labels...')
    attribute_labels = read_vector_format(attribute_labels_file)
    print(next(iter(attribute_labels.items())))

    print('Filtering texture labels...')
    print(f'Total attribute labels: {len(attribute_labels)}')
    texture_names = attribute_names[attribute_names.attribute_type == 1]
    texture_ids = set(texture_names.index)
    texture_labels = {
        img: [attr for attr in attrs if attr in texture_ids]
        for img, attrs in attribute_labels.items()
        if len(set(attrs).intersection(texture_ids)) > 0
    }
    print(f'Total texture labels: {len(texture_labels)}')

    print('Reading eval partitions...')
    test_train_split = read_tuple_format(eval_partitions_file)
    test_train_split = test_train_split.set_index('image_name')
    print(test_train_split.head())

    print('Removing examples without texture labels...')
    category_labels = category_labels[
        category_labels.index.isin(texture_labels)]
    print(f'Examples with texture label: {len(category_labels)}')


    print('Removing duplicate images...')
    unique_images = remove_duplicates(data_dir,
                                      category_labels.index.values)
    unique_category_labels = category_labels[category_labels.index.isin(
        unique_images)]
    unique_texture_labels = {
        img: attrs
        for img, attrs in texture_labels.items() if img in unique_images
    }
    print(f'Unique category labels: {len(unique_category_labels)}')
    print(f'Unique texture labels: {len(unique_texture_labels)}')


    print('Remapping category and texture labels...')
    relevant_category_ids = unique_category_labels.category_label.unique()
    relevant_category_names = category_names[category_names.index.isin(
        relevant_category_ids)]
    print(f'Relevant categories: {len(relevant_category_names)}')

    relevant_texture_ids = set(tex_id
                               for textures in unique_texture_labels.values()
                               for tex_id in textures)
    relevant_texture_names = texture_names[texture_names.index.isin(
        relevant_texture_ids)]
    print(f'Relevant textures: {len(relevant_texture_names)}')

    relevant_category_names['new_id'] = range(len(relevant_category_names) )
    relevant_texture_names['new_id'] = range(len(relevant_texture_names))

    remapped_category_labels = unique_category_labels.category_label.apply(
        lambda cat_id: relevant_category_names.loc[cat_id].new_id)
    remapped_category_labels = pd.DataFrame(remapped_category_labels)

    remapped_texture_labels = {
        img: [
            int(relevant_texture_names.loc[tex_id].new_id)
            for tex_id in textures
        ]
        for img, textures in tqdm(unique_texture_labels.items())
    }

    relevant_category_names = relevant_category_names.set_index('new_id')
    relevant_texture_names = relevant_texture_names.set_index('new_id')

    print(unique_category_labels.join(
        test_train_split).evaluation_status.value_counts())

    print(f'Writing processed output to {data_dir}') 

    Path(data_dir + 'anno_processed').mkdir(exist_ok=True)
    relevant_category_names.to_csv(data_dir +
                                   'anno_processed/category_names.tsv',
                                   index=False,
                                   sep='\t')
    remapped_category_labels.to_csv(data_dir +
                                    'anno_processed/category_labels.tsv',
                                    sep='\t')
    relevant_texture_names.to_csv(data_dir +
                                  'anno_processed/texture_names.tsv',
                                  index=False,
                                  sep='\t')
    save_vector_dict(remapped_texture_labels,
                     data_dir + 'anno_processed/texture_labels.jsonl')

    Path(data_dir + 'eval_processed').mkdir(exist_ok=True)
    test_train_split.to_csv(data_dir + 'eval_processed/eval_partition.tsv',
                            sep='\t')
