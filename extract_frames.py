import sys
sys.path.append('core')

import argparse
import os
import os.path as osp
import cv2
from glob import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from utils import frame_utils
import torch.utils.data as data


class InferenceLoader(data.Dataset):

    def skip_image(self, imfile1, imfile2, output_path):
        dir1 = os.path.dirname(imfile1)
        dir2 = os.path.dirname(imfile2)
        if dir1 != dir2: return True

        check_path = os.path.splitext(output_path)[0] + 'x.jpg'
        if os.path.exists(check_path): return True
        return False

    def __init__(self, args):
        
        images = glob(osp.join(args.path, '**/*.png'), recursive=True) + \
                glob(osp.join(args.path, '**/*.jpg'), recursive=True)
        images = sorted(images)
        print("Found total of %s images" %len(images))

        output_paths = [fn.replace(args.path, args.output_path) for fn in images]
        self.image_list = []

        for img1, img2, output_path in zip(images[:-1], images[1:], output_paths):
            if self.skip_image(img1, img2, output_path): continue
            self.image_list.append([img1, img2, output_path])
        
        print("%s images left to run" %len(self.image_list))

    def __getitem__(self, index):
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        
        w, h = img1.size
        img1 = img1.resize((int(w/2), int(h/2)))
        img2 = img2.resize((int(w/2), int(h/2)))
        
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        return img1, img2, self.image_list[index][2]

    def __len__(self):
        return len(self.image_list)


def viz(imgs, flos, output_paths, batch_size):
    imgs = imgs.permute(0,2,3,1).cpu().numpy()
    flos = flos.permute(0,2,3,1).cpu().numpy()

    for i in range(batch_size):
        img = imgs[i].squeeze()
        flo = flos[i].squeeze()
        output_path = output_paths[i]

        output_path_noext = os.path.splitext(output_path)[0]
        output_dir = os.path.dirname(output_path_noext)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if args.vis:
            flo = flow_viz.flow_to_image(flo)
            img_flo = np.concatenate([img, flo], axis=0)[:, :, [2,1,0]]
            cv2.imwrite(output_path_noext + 'x.jpg', img_flo)
        else:
            cv2.imwrite(output_path_noext + 'x.jpg', flo[:, :, 0])
            cv2.imwrite(output_path_noext + 'y.jpg', flo[:, :, 1])
        print("Done %s" %output_path_noext)


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    dataset = InferenceLoader(args)
    dataloader = data.DataLoader(
        dataset, 
        batch_size=args.batch_size,
        pin_memory=False, 
        shuffle=False, 
        num_workers=args.workers, 
        drop_last=True)

    with torch.no_grad():
        for image1, image2, output_path in dataloader:
            image1 = image1[None].cuda().squeeze(dim=0)
            image2 = image2[None].cuda().squeeze(dim=0)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up, output_path, args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="restore checkpoint")
    parser.add_argument('--path', required=True, help="dataset for evaluation")
    parser.add_argument('--output_path', required=True, help="output directory for flow")
    
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--vis', action='store_true', help='visualise the optical flow')
    
    parser.add_argument('--device', help='device name, default: cuda', default="cuda")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', help='number of cpu workers to load images', type=int, default=2)
    args = parser.parse_args()

    DEVICE = args.device

    demo(args)
