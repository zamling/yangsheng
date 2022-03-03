import os

EDGE_ROOT = '/data1/zem/MVSS-Net/data/edge_mask'
MASK_ROOT = '/data1/zem/MVSS-Net/data/archive/CASIA2/gt'
IMG_ROOT = '/data1/zem/MVSS-Net/data/archive/CASIA2/Tp'

def generate_data():
    with open('/data1/zem/MVSS-Net/data/CASIAv2.txt', 'r') as f:
        data = f.readlines()
    lines = []
    for _data in data:
        img, mask, label = _data.strip().split(' ')
        img_name = img.split('/')[-1]
        mask_name = mask.split('/')[-1]
        new_img = os.path.join(IMG_ROOT, img_name)
        new_mask = os.path.join(MASK_ROOT, mask_name)
        new_edge = os.path.join(EDGE_ROOT, mask_name)
        new_line = new_img + ' ' + new_mask + ' ' + new_edge + ' ' + label + '\n'
        lines.append(new_line)
    with open('/data1/zem/MVSS-Net/data/mydata.txt', 'w') as f:
        f.writelines(lines)

def check_data():
    with open('/data1/zem/MVSS-Net/data/mydata.txt', 'r') as f:
        data = f.readlines()




