import numpy as np
import cv2
import argparse

from JBF import Joint_bilateral_filter


def main(args):
    print('Processing %s ...'%args.image_path)
    # Load image
    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # Set a guidance image
    guidance = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if args.gray else img_rgb

    # Hyperparameters
    sigma_s = args.Ss
    sigma_r = args.Sr

    # Joint bilateral filter
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    result = JBF.joint_bilateral_filter(img_rgb, guidance).astype(np.uint8)

    # Save result
    if args.save_path == None:
        save_path = args.image_path[:-4] + '_filtered.png'
    else:
        save_path = args.save_path

    cv2.imwrite(save_path, cv2.cvtColor(result,cv2.COLOR_BGR2RGB))

    print('Result saved to %s'%save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', required=True, help='path to input image')
    parser.add_argument('--save_path', default=None, help='path to save result')
    parser.add_argument('--Ss', default=2, type=int, help='Standard deviation of spatial kernel')
    parser.add_argument('--Sr', default=0.1, type=float, help='Standard deviation of range kernel')
    parser.add_argument('--gray', action='store_true', help='Use grayscale image as guidance image')
    args = parser.parse_args()
    main(args)