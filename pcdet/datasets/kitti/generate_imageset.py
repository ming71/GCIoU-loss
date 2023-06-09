import os

trainset = r'/workspace/OpenPCDet/data/kitti/ImageSets/train.txt'
testset  = r'/workspace/OpenPCDet/data/kitti/ImageSets/test.txt'
tr_im_dir = r'/workspace/OpenPCDet/data/kitti/training/image_2'
ts_im_dir = r'/workspace/OpenPCDet/data/kitti/testing/image_2'

with open(trainset, 'w') as f:
    files = os.listdir(tr_im_dir)
    names = [x.split('.')[0] for x in files]
    s = '\n'.join(names)
    f.write(s)


with open(testset, 'w') as f:
    files = os.listdir(ts_im_dir)
    names = [x.split('.')[0] for x in files]
    s = '\n'.join(names)
    f.write(s)