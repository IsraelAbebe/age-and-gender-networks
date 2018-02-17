import os
import cv2


def convert_graysale(f):
    '''
    :param f: file name
    :return: returns the grayscaled and scaled version of the image
    '''
    image = cv2.imread(f,0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f, gray_image)
    return gray_image


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
            if ".jpg" in f:
                path =  tail + "/" + f
                print(count , f ,path)
                count += 1


grayscale_save()