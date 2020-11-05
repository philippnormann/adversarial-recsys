import gdown
import zipfile
from pathlib import Path

# DeepFashion: Attribute Prediction
# https://drive.google.com/open?id=0B7EVK8r0v71pQ2FuZ0k0QnhBQnc

data_dir = 'data/DeepFashion/'
files = {
    'attr_cloth': {
        'url': 'https://drive.google.com/uc?id=0B7EVK8r0v71pYnBKQVBOaHR1WWs',
        'file': 'anno/list_attr_cloth.txt'
    },
    'attr_img': {
        'url': 'https://drive.google.com/uc?id=0B7EVK8r0v71pWXE4QWotX2hxQ1U',
        'file': 'anno/list_attr_img.txt'
    },
    'category_cloth': {
        'url': 'https://drive.google.com/uc?id=0B7EVK8r0v71pWnFiNlNGTVloLUk',
        'file': 'anno/list_category_cloth.txt'
    },
    'category_img': {
        'url': 'https://drive.google.com/uc?id=0B7EVK8r0v71pTGNoWkhZeVpzbFk',
        'file': 'anno/list_category_img.txt'
    },
    'eval_partition': {
        'url': 'https://drive.google.com/uc?id=0B7EVK8r0v71pdS1FMlNreEwtc1E',
        'file': 'eval/list_eval_partition.txt'
    },
    'img': {
        'url': 'https://drive.google.com/uc?id=0B7EVK8r0v71pa2EyNEJ0dE9zbU0',
        'file': 'img.zip'
    }
}

for name, file_config in files.items():
    gdrive_url = file_config['url']
    out_path = Path(data_dir + file_config['file'])
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        print(f'Dowloading {name} to {out_path}...')
        gdown.download(gdrive_url, str(out_path))
    else:
        print(f'Skipping already downloaded {name}...')
    if str(out_path).endswith('.zip'):
        if not (out_dir / name).exists():
            print(f'Unzipping {name}...')
            with zipfile.ZipFile(out_path, "r") as zip_ref:
                zip_ref.extractall(out_dir)
        else:
            print(f'Skipping already unzipped {name}...')

print('All done!')
