import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s

        size = self.wndw_size//2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        self.gaussian_filter = (-(x*x+y*y)/(self.sigma_s*self.sigma_s*2)).reshape(-1,)
        self.rr = self.sigma_r*self.sigma_r*(-2)
    

    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE)

        guidance = guidance/255
        padded_guidance = padded_guidance/255

        if len(guidance.shape) > 2:
            win_size = self.wndw_size*self.wndw_size
            x_len = img.shape[1]
            y_len = img.shape[0]
            img_vec_R = guidance[:,:,0]
            img_vec_G = guidance[:,:,1]
            img_vec_B = guidance[:,:,2]
            
            output = np.zeros(img.shape)
            w = (padded_guidance[:y_len,:x_len,0]-img_vec_R)**2 + (padded_guidance[:y_len,:x_len,1]-img_vec_G)**2 + (padded_guidance[:y_len,:x_len,2]-img_vec_B)**2
            w = np.exp(w/self.rr+self.gaussian_filter[0])
            w_total = w

            output[:,:,0] += w*padded_img[:y_len, :x_len, 0]
            output[:,:,1] += w*padded_img[:y_len, :x_len, 1]
            output[:,:,2] += w*padded_img[:y_len, :x_len, 2]

            for i in range(1, win_size):
                x = i%self.wndw_size
                y = i//self.wndw_size
                w = (padded_guidance[y:y+y_len,x:x+x_len,0]-img_vec_R)**2 + (padded_guidance[y:y+y_len,x:x+x_len,1]-img_vec_G)**2 + (padded_guidance[y:y+y_len,x:x+x_len,2]-img_vec_B)**2
                w = np.exp(w/self.rr+self.gaussian_filter[i])
                w_total += w

                output[:,:,0] += w*padded_img[y:y+y_len, x:x+x_len, 0]
                output[:,:,1] += w*padded_img[y:y+y_len, x:x+x_len, 1]
                output[:,:,2] += w*padded_img[y:y+y_len, x:x+x_len, 2]

            output[:,:,0] /= w_total
            output[:,:,1] /= w_total
            output[:,:,2] /= w_total

        else:
            win_size = self.wndw_size*self.wndw_size
            x_len = img.shape[1]
            y_len = img.shape[0]
            
            w = np.exp(((padded_guidance[:y_len, :x_len]-guidance)**2/self.rr)+self.gaussian_filter[0])
            w_total = w

            output = np.zeros(img.shape)
            output[:,:,0] += w*padded_img[:y_len, :x_len, 0]
            output[:,:,1] += w*padded_img[:y_len, :x_len, 1]
            output[:,:,2] += w*padded_img[:y_len, :x_len, 2]

            for i in range(1, win_size):
                x = i%self.wndw_size
                y = i//self.wndw_size
                w = np.exp(((padded_guidance[y:y+y_len, x:x+x_len]-guidance)**2/self.rr)+self.gaussian_filter[i])

                w_total += w
                output[:,:,0] += w*padded_img[y:y+y_len, x:x+x_len, 0]
                output[:,:,1] += w*padded_img[y:y+y_len, x:x+x_len, 1]
                output[:,:,2] += w*padded_img[y:y+y_len, x:x+x_len, 2]

            output[:,:,0] /= w_total
            output[:,:,1] /= w_total
            output[:,:,2] /= w_total

        return np.clip(output, 0, 255).astype(np.uint8)