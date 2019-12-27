import os
import cv2


def convert_graysale(f):
    '''
    :param f: file name
    :return: returns the grayscaled and scaled version of the image
    '''
    image = cv2.imread(f,0)
    result = cv2.imwrite(f, image)
    return result


def grayscale_save(fi):
    '''
    :param f: file path
    :return: grayscales all images in the directory
    '''
    count = 0
    for cur, _dirs, files in os.walk(fi):
        head, tail = os.path.split(cur)
        while head:
            head, _tail = os.path.split(head)

        for f in files:
            if ".jpg" in f:
                path =  fi + '/'+tail + "/" + f
                a = convert_graysale(path)
                print(count , f ,a)
                count += 1


grayscale_save('data/images/train')