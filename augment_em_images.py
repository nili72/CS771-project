import os
import cv2

def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class my_img:
    def __init__(self, file_name, file_dir):
        self.file_path = os.path.join(file_dir, file_name)
        self.file_name = file_name
        self.img_name = file_name.split(".")[0]
        self.img = cv2.imread(self.file_path)
        self.imsize = self.img.shape
        self.im_height = self.imsize[0]
        self.im_width = self.imsize[1]
        self.img_crops = {}
    
    def crop_img(self, crop_size, out_dir):
        '''crop image in squares given crop size, then save'''
        cnt = 0
        for row in range(0, self.im_height, crop_size):
            for col in range(0, self.im_width, crop_size):
                img_crop = self.img[row:row+crop_size, col:col+crop_size, :]
                self.img_crops[cnt] = img_crop
                # save crop
                out_name = f"{self.img_name}_crop_{cnt}.jpg"
                out_path = os.path.join(out_dir, out_name)
                cv2.imwrite(out_path,img_crop)
                cnt +=1

    def crop_rotate(self, out_dir, degs = 3):
        '''rotate crops 0->90, 1->180, 2->270, 3->all'''
        degrees = ["90", "180", "270"]
        for cnt, crop in self.img_crops.items():
            rot90 = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            rot180 = cv2.rotate(crop, cv2.ROTATE_180)
            rot270 = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rot_imgs = {0:rot90, 1:rot180, 2:rot270}
            if degs == 0:
                out_name = f"{self.img_name}_crop_{cnt}_{degrees[degs]}.jpg"
                out_path = os.path.join(out_dir, out_name)
                cv2.imwrite(out_path, rot_imgs[degs])
            elif degs == 1:
                out_name = f"{self.img_name}_crop_{cnt}_{degrees[degs]}.jpg"
                out_path = os.path.join(out_dir, out_name)
                cv2.imwrite(out_path, rot_imgs[degs])
            elif degs == 2:
                out_name = f"{self.img_name}_crop_{cnt}_{degrees[degs]}.jpg"
                out_path = os.path.join(out_dir, out_name)
                cv2.imwrite(out_path, rot_imgs[degs])
            elif degs == 3:
                for dg, rimg in rot_imgs.items():
                    out_name = f"{self.img_name}_crop_{cnt}_{degrees[dg]}.jpg"
                    out_path = os.path.join(out_dir, out_name)
                    cv2.imwrite(out_path, rimg)
    
    def crop_flip(self, out_dir, ftype = 2):
        '''flip crops 0 vertical, 1 horizontal, 2 both'''
        for cnt, crop in self.img_crops.items():
            flip = ["vflip", "hflip"]
            hflip = cv2.flip(crop, 1)
            vflip = cv2.flip(crop, 0)
            fliped_im = {0: vflip, 1: hflip}
            if ftype ==0:
                out_name = f"{self.img_name}_crop_{cnt}_{flip[ftype]}.jpg"
                out_path = os.path.join(out_dir, out_name)
                cv2.imwrite(out_path, fliped_im[ftype])
            elif ftype ==1:
                out_name = f"{self.img_name}_crop_{cnt}_{flip[ftype]}.jpg"
                out_path = os.path.join(out_dir, out_name)
                cv2.imwrite(out_path, fliped_im[ftype])
            elif ftype == 2:
                for ft, fimg in fliped_im.items():
                    out_name = f"{self.img_name}_crop_{cnt}_{flip[ft]}.jpg"
                    out_path = os.path.join(out_dir, out_name)
                    cv2.imwrite(out_path, fimg)
        def crop_rotate_flp():
            pass

'''
Directory settinngs
dir_path:
    augment_em_images.py (change dir_path if files and script and in different locations)
    data: contains the original images
    augmented_data: where augmented images will be saved
'''
dir_path = os.getcwd()
# dir_path = data_directory # change for our prefered directory
data_path = os.path.join(dir_path, "data") # input
out_path =  os.path.join(dir_path, "augmented_data") # output
mk_dir(out_path)

'''
Augument settings
Flip
     0-> vertical, 1-> horizontal, 2-> both, None -> False
Rotate
    0-> 90, 1-> 180, 2-> 270, 3-> all, None -> False
'''
crop_size = 256
flip = 2
rotate = 3

# list images in data folder
file_list = os.listdir(data_path)

for filename in file_list:
    # read image
    image = my_img(filename, data_path)
    # crop image
    image.crop_img(crop_size, out_path)
    # flip crops
    if flip is not False:
        image.crop_flip(out_path, flip)
    # rotate crops
    if rotate is not False:
        image.crop_rotate(out_path, rotate)
    # finish 