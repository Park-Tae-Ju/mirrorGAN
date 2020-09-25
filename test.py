from model import CAPTION_CNN, CAPTION_RNN
import torch
import argparse
from cfg.config import cfg, cfg_from_file
import pprint
from PIL import Image
from torchvision import transforms
from datasets import TextDataset
from torch.autograd import Variable
from glob import glob
import os
import os.path as osp
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_attn2.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    args = parser.parse_args()
    return args


def prepare_data(data):
    imgs, captions, captions_lens, class_ids, keys = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys]

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    # image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image

args = parse_args()
if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)

if args.data_dir != '':
    cfg.DATA_DIR = args.data_dir

split_dir, bshuffle = 'train', True
if not cfg.TRAIN.FLAG:
    # bshuffle = False
    split_dir = 'test'
print('Using config:')
pprint.pprint(cfg)


# Get data loader
imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
## data preprocessing
# image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406),
    #                      (0.229, 0.224, 0.225))
])
# load text data
dataset = TextDataset(cfg.DATA_DIR, split_dir,
                      base_size=cfg.TREE.BASE_SIZE,
                      transform=transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load caption model
caption_cnn = CAPTION_CNN(cfg.CAP.embed_size)
caption_rnn = CAPTION_RNN(cfg.CAP.embed_size, cfg.CAP.hidden_size * 2, dataset.n_words, cfg.CAP.num_layers)
caption_cnn.to(device)
caption_rnn.to(device)

caption_cnn.load_state_dict(torch.load(cfg.CAP.caption_cnn_path, map_location=lambda storage, loc: storage))
caption_rnn.load_state_dict(torch.load(cfg.CAP.caption_rnn_path, map_location=lambda storage, loc: storage))

## inference
# load image dataset
ROOT_DIR = os.getcwd()
# DATA_DIR = osp.join(ROOT_DIR, 'data', 'output', 'bird', 'Model', 'netG', 'valid', 'single')
DATA_DIR = 'data/birds/CUB_200_2011/images'
OUTPUT_DIR = osp.join(ROOT_DIR, 'output')
if not osp.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
data_infos = []
for dir_path in glob(osp.join(DATA_DIR, '*'), recursive=True):
    if osp.isdir(dir_path):
        image_path = []
        for f in glob(osp.join(dir_path, '*.jpg')):
            image_path.append(f)
        data_infos.append(
            {
                "folder_path": dir_path,
                "folder_name": osp.basename(dir_path),
                "file_name": ['_'.join(osp.basename(p).split('_')[:-1]) for p in image_path],
                "image_paths": image_path
            }
        )

for info in tqdm(data_infos):
    for i, image_path in enumerate(info['image_paths']):
        image = load_image(image_path, transform=transform)
        image_tensor = image.to(device)
        features = caption_cnn(image_tensor, subset='test')
        sampled_ids = caption_rnn.sample(features)
        sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = dataset.ixtoword[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
        print(sentence)
        text = sentence.replace('<start>', '').replace('<end>', '').strip()
        if not osp.exists(osp.join(OUTPUT_DIR, info['folder_name'])):
            os.mkdir(osp.join(OUTPUT_DIR, info['folder_name']))

        with open(osp.join(OUTPUT_DIR, info['folder_name'], '{}.txt'.format(info['file_name'][i])), 'w') as f:
            f.write(text)