import argparse
import glob
import os
import random
import tqdm
import numpy as np
from shutil import copyfile
from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # TODO: Implement function
    files = glob.glob(os.path.join(source, 'processed/*'))

    idxs = np.array(list(range(len(files))))

    np.random.shuffle(idxs)
    for i in ['train', 'val', 'test']:
        os.makedirs(os.path.join(destination, i), exist_ok = True)
    with tqdm(total=len(files) ) as pbar:
        for ii in range(len(files)):
            
            file = files[idxs[ii]]
            if ii <=len(files)*0.7:
                copyfile(file, file.replace('processed', 'train'))
            elif ii <=len(files)*0.85:
                copyfile(file, file.replace('processed', 'val'))
            else:
                copyfile(file, file.replace('processed', 'test'))
            pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)