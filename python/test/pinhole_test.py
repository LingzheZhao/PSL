import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import imageio
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from colormap_image import colormap_image
from pypsl import PSL


def to8b(x):
    """Converts a torch tensor to 8 bit"""
    return (255 * torch.clamp(x, min=0, max=1)).to(torch.uint8)

def read_images(file_list_path: Path) -> List[torch.Tensor]:
    assert file_list_path.exists(), f"{file_list_path} does not exist."
    parent_dir = file_list_path.parent

    with open(file_list_path, 'r') as f:
        image_files = f.readlines()
    image_files = [parent_dir / x.strip() for x in image_files]

    images = []
    for p in tqdm.tqdm(image_files):
        img = imageio.v3.imread(str(p.resolve()))
        images.append(img)
    images = [torch.tensor(img) for img in images]
    print(f"Read {len(images)} images.")
    return images


def read_poses(file_path: Path) -> Dict[int, NDArray]:
    assert file_path.exists(), f"{file_path} does not exist."
    # read file into buffer, replace \n with space
    buffer = file_path.read_text().replace('\n', ' ')
    # split buffer by space
    buffer = buffer.split()

    num_cameras = -1
    poses = {}
    for i, s in enumerate(buffer):
        if i == 0:
            num_cameras = int(s)
            continue
        # read camera poses, each pose has 13 values, 1 for index, 12 for pose matrix
        if i % 13 == 1:
            index = int(s)
            pose = buffer[i + 1:i + 13]
            pose = [float(x) for x in pose]
            # to numpy array
            pose = np.array(pose).reshape(3, 4)
            pose = np.vstack([pose, np.array([0, 0, 0, 1])])
            poses[index] = pose
    assert len(poses.keys()) == num_cameras, "Number of cameras does not match."
    print(f"Read {len(poses.keys())} poses.")
    return poses


def main(config_path: Path, data_dir: Path):
    configs = json.loads(config_path.read_text())
    psl = PSL(configs)
    images = read_images(data_dir / "images.txt")
    poses = read_poses(data_dir / "model-0-cams.txt")
    for i, image in enumerate(images):
        psl.sweeper.add_frame(i, image, poses[i])

    frame_ids = psl.sweeper.get_frame_ids()
    print(f"frame_ids: {frame_ids}")

    depth_image = psl.sweeper.process(2)
    print(f"depth_image: {depth_image.shape}")
    # fill Nan with 0
    depth_image[torch.isnan(depth_image)] = 0
    print(f"min: {depth_image.min()}, max: {depth_image.max()}")
    depth_image = colormap_image(depth_image[None, ...])
    # C,H,W -> H,W,C
    depth_image = depth_image.permute(1, 2, 0)
    print(f"min: {depth_image.min()}, max: {depth_image.max()}")
    cv2.imshow("depth_image", depth_image.cpu().numpy())
    cv2.waitKey(0)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 test/pinhole_test.py <config_path> <data_dir>")
        sys.exit(1)
    config_path = Path(sys.argv[1])
    data_dir = Path(sys.argv[2])
    assert config_path.suffix == ".json", "config_path should be a json file."
    assert data_dir.exists(), "data_dir does not exist."

    main(config_path, data_dir)
