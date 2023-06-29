import numpy as np
import cv2
import argparse
import os

from JBF import Joint_bilateral_filter


def main(args):
    print('Processing %s ...'%args.image_path)
    # Hyperparameters
    sigma_s = args.Ss
    sigma_r = args.Sr
    R, G, B = args.R, args.G, args.B

    # Load image
    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # Set a guidance image
    guidance = R*img_rgb[:,:,0] + G*img_rgb[:,:,1] + B*img_rgb[:,:,2]

    # Joint bilateral filter
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    # Guided by the original image
    self_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    # Guided by the grayscale image
    gray_out = JBF.joint_bilateral_filter(img_rgb, guidance).astype(np.uint8)
    
    # Calculate the cost
    cost = np.sum(np.abs(self_out.astype('int32')-gray_out.astype('int32'))) / (img.shape[0]*img.shape[1]*3)
    print("Cost: %f"%cost)

    # Save result
    cv2.imwrite(os.path.join(args.save_dir, 'RGB_filtered.png'), cv2.cvtColor(self_out,cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(args.save_dir, 'gray_filtered.png'), cv2.cvtColor(gray_out,cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(args.save_dir, 'gray.png'), guidance)

    print('Result saved to', args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', required=True, help='path to input image')
    parser.add_argument('--save_dir', required=True, help='path to save result')
    parser.add_argument('--Ss', default=2, type=int, help='Standard deviation of spatial kernel')
    parser.add_argument('--Sr', default=0.1, type=float, help='Standard deviation of range kernel')
    parser.add_argument('--R', default=0.3, type=float, help='Weight of red channel')
    parser.add_argument('--G', default=0.4, type=float, help='Weight of green channel')
    parser.add_argument('--B', default=0.3, type=float, help='Weight of blue channel')
    args = parser.parse_args()
    main(args)