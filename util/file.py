import os
import shutil


def make_dir(target_dir: str)->None:
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

def clear_dir(target_dir: str)->None:
    if not os.path.exists(target_dir):
        raise Exception(f'{target_dir} 目录不存在')
    else:
        shutil.rmtree(target_dir)

