""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

from pathlib import Path

import fire
import os
import json
import lmdb
import cv2
import numpy as np
from natsort import natsorted
from PIL import Image


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    for i in range(nSamples):
        imagePath, label = datalist[i].strip('\n').split('\t')
        imagePath = os.path.join(inputPath, imagePath)

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)
    env.close()


class _LPMeta:
    __slots__ = [
        "label",  # LP label
        "pos"  # [x0 y0 x1 y1]
    ]
    def is_valid(self):
        return self.label is not None and self.pos is not None


def createDatasetALPR(inputPath, outputPath, checkValid=True):
    """Create LMDB dataset of UFPR-ALPR for training and evaluation.
    UFPR-ALPR has following directory format;
        .
        ├── training
            ├── track0001
                ├── track0001[01].png
                ├── track0001[01].txt
                ├── track0001[02].png
                ├── track0002[02].txt
                ...
            ├── track0002
            ...
        ├── validation
            ├── track0061
            ..
        └── testing
            ├── track0091
            ...

    Example `.txt` metadata file;
        ---------------------------------
        camera: GoPro Hero4 Silver
        position_vehicle: 835 310 140 312
                type: motorcycle
                make: Yamaha
                model: XTZ
                year: 2017
        plate: BBO-8514
        position_plate: 889 506 42 35
                char 1: 895 515 8 12
                char 2: 906 515 8 12
                char 3: 916 515 9 12
                char 4: 894 527 8 11
                char 5: 902 527 8 11
                char 6: 912 527 4 11
                char 7: 918 527 8 11
        ---------------------------------
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        checkValid : if true, check the validity of every image
    """
    def _parse_meta(metaPath):
        meta = _LPMeta()
        with open(metaPath, 'r') as fd:
            for _, line in enumerate(fd):
                if line.startswith('plate:'):
                    meta.label = line.split(':')[1].strip()
                if line.startswith('position_plate:'):
                    meta.pos = map(int, line.split(':')[1].strip().split())
        return meta

    class _Dataset:
        def __init__(self, root):
            self.image_path_list = list()
            for dirpath, dirnames, filenames in os.walk(root):
                for name in filenames:
                    _, ext = os.path.splitext(name)
                    if ext.lower() in ['.jpg', '.jpeg', '.png']:
                        self.image_path_list.append(os.path.join(dirpath, name))

            self.image_path_list = natsorted(self.image_path_list)
            self.nSamples = len(self.image_path_list)

        def __len__(self):
            return self.nSamples

        def __getitem__(self, index):
            imagePath = self.image_path_list[index]
            prepath, imageExt = os.path.splitext(imagePath)
            metaPath = prepath + ".txt"
            cropPath = prepath + ".crop"  # avoid using img extension

            if not os.path.exists(imagePath):
                print('%s does not exist' % imagePath)
                return (None, None, imagePath)

            # parse LP label from metafile
            try:
                meta = _parse_meta(metaPath)
                if not meta.is_valid:
                    print('invalid meta: %s' % metaPath)
                    return (None, None, imagePath)
            except Exception as e:
                print('error occured for meta', metaPath, str(e))
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('label parsing error: %s\n' % metaPath)
                return (None, None, imagePath)

            # crop image
            try:
                image = Image.open(imagePath)
            except IOError:
                print('Corrupted image: %s' % imagePath)
                return (None, None, imagePath)
            x, y, w, h = meta.pos
            box = (x, y, x + w, y + h)
            cropped_image = image.crop(box)
            cropped_image.save(cropPath, format=imageExt[1:])

            with open(cropPath, 'rb') as f:
                img = f.read()
            return (img, meta, imagePath)


    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)  # 1024GB
    cache = {}
    cnt = 1

    dataset = _Dataset(root=inputPath)
    nSamples = len(dataset)

    for (imageBin, imageMeta, imagePath) in dataset:
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except Exception as e:
                print('error occured for image', imagePath, str(e))
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('image data occured error: \n' % imagePath)
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = imageMeta.label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1

    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)  # flush cache
    print('Created dataset with %d samples' % nSamples)
    env.close()


def createDatasetBbox(inputPath, outputPath, checkValid=True):
    """
    ├── train5000-5500-crop
        ├── _lp.json
        ├── 00000002-roi1--2018-01-25-08-23-59-b0.jpeg
        ├── 00000002-roi1--2018-01-25-09-05-59-b0.jpeg
        ├── ...
    ├── train3462-3962-crop
        ├── _lp.json
        ├── ...
    ...
    """
    class _Dataset:
        def __init__(self, root):
            self.root = root
            self.image_path_list = list()
            for dirpath, dirnames, filenames in os.walk(root):
                for name in filenames:
                    _, ext = os.path.splitext(name)
                    if ext.lower() in ['.jpg', '.jpeg', '.png']:
                        self.image_path_list.append(os.path.join(dirpath, name))

            with open(os.path.join(root, '_lp.json')) as fd:
                self._meta = json.load(fd)

            self.image_path_list = natsorted(self.image_path_list)
            self.nSamples = len(self.image_path_list)

        def __len__(self):
            return self.nSamples

        def __getitem__(self, index):
            imagePath = self.image_path_list[index]

            if not os.path.exists(imagePath):
                print('%s does not exist' % imagePath)
                return (None, None, imagePath)

            meta = _LPMeta()
            meta_key = f"{os.path.basename(self.root)}{imagePath.replace(self.root, '')}"
            meta.label = self._meta[meta_key]

            with open(imagePath, 'rb') as f:
                img = f.read()
            return (img, meta, imagePath)

    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)  # 1024GB
    cache = {}
    nSamplesTotal = 0
    cnt = 1
    precnt = 1

    for dirpath, dirnames, filenames in os.walk(inputPath):
        for dirname in dirnames:
            rootpath = os.path.join(dirpath, dirname)
            print(f"reading input path: {rootpath}")
            dataset = _Dataset(root=rootpath)
            nSamples = len(dataset)
            print(f"{dirname} dataset size: {nSamples}")

            for (imageBin, imageMeta, imagePath) in dataset:
                if checkValid:
                    try:
                        if not checkImageIsValid(imageBin):
                            print('%s is not a valid image' % imagePath)
                            continue
                    except Exception as e:
                        print('error occured for image', imagePath, str(e))
                        with open(outputPath + '/error_image_log.txt', 'a') as log:
                            log.write('image data occured error: \n' % imagePath)
                        continue

                imageKey = 'image-%09d'.encode() % cnt
                labelKey = 'label-%09d'.encode() % cnt
                cache[imageKey] = imageBin
                cache[labelKey] = imageMeta.label.encode()

                if cnt % 1000 == 0:
                    writeCache(env, cache)  # flush cache
                    cache = {}
                    print('Written %d / %d' % (cnt-precnt, nSamples))
                cnt += 1
            writeCache(env, cache)  # flush cache
            cache = {}
            print('Written %d / %d' % (cnt-precnt, nSamples))
            nSamplesTotal += (cnt-precnt)
            precnt = cnt

    cache['num-samples'.encode()] = str(nSamplesTotal).encode()
    writeCache(env, cache)  # flush cache
    print('Created dataset with %d samples' % nSamplesTotal)
    env.close()


# def test_crop():
#     src_path = '/Users/evren/ds/track0052[01].png'
#     dst_path = '/Users/evren/ds/track0052[01]_crop.png'
#     position_plate = '843 712 102 36'
#     pos = map(int, position_plate.split())
#     x, y, w, h = pos
#     print(x, y, w, h)
#
#     try:
#         image = Image.open(src_path)
#     except IOError:
#         print('Corrupted image: %s' % src_path)
#     box = (x, y, x+w, y+h)
#     cropped_image = image.crop(box)
#     cropped_image.save(dst_path)


if __name__ == '__main__':
    fire.Fire(createDatasetBbox)

# python3 create_lmdb_dataset --inputPath train-crop --outputPath train-lmdb
# Created dataset with 3885 samples

# python3 create_lmdb_dataset --inputPath validation-crop --outputPath validation-lmdb
# Created dataset with 1332 samples

# python3 create_lmdb_dataset --inputPath test-crop --outputPath test-lmdb
# Created dataset with 667 samples
