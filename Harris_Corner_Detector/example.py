import numpy as np
import cv2
import argparse
from HCD import Harris_corner_detector


def main(args):
    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    HCD = Harris_corner_detector(args.threshold)
    response = HCD.detect_harris_corners(img_gray)
    result = HCD.post_processing(response)

    for r in result:
        cv2.rectangle(img, (r[1]-1,r[0]-1), (r[1]+1,r[0]+1), (0,0,255), 1)
    
    if args.save_path is None:
        save_path = args.image_path[:-4]+'_corner_{}'.format(int(args.threshold))+'.png'
    else:
        save_path = args.save_path

    cv2.imwrite(save_path, img)

    print('Result saved to %s'%save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main function of Harris corner detector')
    parser.add_argument('--threshold', default=100., type=float, help='threshold value to determine corner')
    parser.add_argument('--image_path', required=True, help='path to input image')
    parser.add_argument('--save_path', default=None, help='path to save result')
    args = parser.parse_args()
    main(args)