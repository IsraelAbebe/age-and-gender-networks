import os
import cv2


def grayscale_save(f):
    '''
    :param f: file path
    :return: grayscales all images in the directory
    '''
    count = 0
    for cur, _dirs, files in os.walk(f):
        head, tail = os.path.split(cur)
        while head:
            head, _tail = os.path.split(head)

        for f in files:
            print(count , f)
            count += 1


grayscale_save()