import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from PIL import Image
#import matplotlib.pyplot as plt

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

T0_MEAN_VALUE = (107.800,117.692,119.979)
T1_MEAN_VALUE = (110.655,117.107,119.135)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

palette = [0, 0, 0,255,255,255]
'''''''''
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)
'''''''''

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_pascal_labels():
    return np.asarray([[0,0,0],[255,255,255]])

def decode_segmap(temp, plot=False):

    label_colours = get_pascal_labels()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 2):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    #rgb = np.resize(rgb,(321,321,3))
    if plot:
        #plt.imshow(rgb)
        #plt.show()
        pass
    else:
        return rgb

#### source dataset is only avaiable by sending an email request to author #####
#### upon request is shown in http://ghsi.github.io/proj/RSS2016.html ####
#### more details are presented in http://ghsi.github.io/assets/pdfs/alcantarilla16rss.pdf ###
class Dataset(Dataset):

    def __init__(self,img_path,label_path,file_name_txt_path,split_flag, transform=True, transform_med = None):

        self.label_path = label_path
        self.img_path = img_path
        #self.img2_path = img2_path
        self.img_txt_path = file_name_txt_path
        self.imgs_path_list = np.loadtxt(self.img_txt_path,dtype=str)
        self.flag = split_flag
        self.transform = transform
        self.transform_med = transform_med
        self.img_label_path_pairs = self.get_img_label_path_pairs()

    def get_img_label_path_pairs(self):
        img_label_pair_list = {}
        if self.flag == 'train':
            for idx, did in enumerate(open(self.img_txt_path)):
                try:
                    # print('did;',did)
                    image1_name, image2_name, mask_name, roi_name = did.strip("\n").split(' ')
                    # print('image1_name,image2_name,mask_name, roi_name:', image1_name,image2_name,mask_name, roi_name)
                except ValueError:  # Adhoc for test.
                    image_name = mask_name = did.strip("\n")
                extract_name = image1_name[image1_name.rindex('/') + 1: image1_name.rindex('.')]
                # print extract_name

                # print("img_path: ", self.img_path)
                # print("image1_name: ", image1_name)
                # print("image2_name: ", image2_name)
                img1_file = os.path.join(self.img_path, image1_name)
                img2_file = os.path.join(self.img_path, image2_name)
                lbl_file = os.path.join(self.label_path, mask_name)
                roi_file = os.path.join(self.img_path, roi_name)
                # print('mask_name:', mask_name)
                # print("img1_file: ", img1_file)
                # print("img2_file: ", img2_file)
                # print("lbl_file: ", lbl_file)
                # print('roi_file:', roi_file)

                img_label_pair_list.setdefault(idx, [img1_file, img2_file, lbl_file, roi_file,
                                                     image2_name])  # 如果字典中包含有给定键，则返回该键对应的值，否则返回为该键设置的值。

        if self.flag == 'val':
            self.label_ext = '.png'
            for idx, did in enumerate(open(self.img_txt_path)):
                try:
                    image1_name, image2_name, mask_name, roi_name = did.strip("\n").split(' ')
                except ValueError:  # Adhoc for test.
                    image_name = mask_name = did.strip("\n")
                extract_name = image1_name[image1_name.rindex('/') + 1: image1_name.rindex('.')]
                # print extract_name
                img1_file = os.path.join(self.img_path, image1_name)
                img2_file = os.path.join(self.img_path, image2_name)
                lbl_file = os.path.join(self.label_path, mask_name)
                roi_file = os.path.join(self.img_path, roi_name)
                img_label_pair_list.setdefault(idx, [img1_file, img2_file, lbl_file, roi_file, image2_name])

        if self.flag == 'test':

            for idx, did in enumerate(open(self.img_txt_path)):
                image1_name, image2_name = did.strip("\n").split(' ')
                img1_file = os.path.join(self.img_path, image1_name)
                img2_file = os.path.join(self.img_path, image2_name)
                img_label_pair_list.setdefault(idx, [img1_file, img2_file, None, None, image2_name])

        return img_label_pair_list

    def data_transform(self, img1,img2,lbl, roi):
        img1 = img1[:, :, ::-1]  # RGB -> BGR
        img1 = img1.astype(np.float32)
        img1 -= T0_MEAN_VALUE
        img1 = img1.transpose(2, 0, 1) # H×W×C -> C×H×W
        #print("unique img1: ", np.unique(img1))
        img1 = img1/128.0 - 1.0

        #img1 = torch.from_numpy(img1).float()

        img2 = img2[:, :, ::-1]  # RGB -> BGR
        img2 = img2.astype(np.float32)
        img2 -= T1_MEAN_VALUE
        img2 = img2.transpose(2, 0, 1)
        #print("unique img2: ", np.unique(img2))
        img2 = img2 / 128.0 - 1.0

        #img2 = torch.from_numpy(img2).float()
        crop_width = 256
        _, h, w = img1.shape

        #print("h: ", h)
        #print("w: ", w)

        #print("lbl.shape: ", lbl.shape)
        #print("roi.shape: ", roi.shape)

        x_l = np.random.randint(0, w - crop_width)
        x_r = x_l + crop_width
        y_l = np.random.randint(0, h - crop_width)
        y_r = y_l + crop_width

        input_ = torch.from_numpy(np.concatenate((img1[:, y_l:y_r, x_l:x_r], img2[:, y_l:y_r, x_l:x_r]), axis=0))

        lbl = lbl.transpose(2, 0, 1)

        mask_ = torch.from_numpy(lbl[:, y_l:y_r, x_l:x_r]).long()
        roi_ = torch.from_numpy(roi[ y_l:y_r, x_l:x_r]).long()

        #if self.flag != 'test':
        #  lbl = torch.from_numpy(lbl).long()
        #  lbl = lbl.unsqueeze(0)
          #lbl_reverse = torch.from_numpy(lbl_reverse).long()
        return input_, mask_, roi_

    def __getitem__(self, index):

        img1_path,img2_path,label_path,roi_path, filename = self.img_label_path_pairs[index]
        ####### load images #############
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        #print filename
        height,width,_ = np.array(img1,dtype= np.uint8).shape
        if self.transform_med != None:
           img1 = self.transform_med(img1)
           img2 = self.transform_med(img2)
        img1 = np.array(img1,dtype= np.uint8)
        img2 = np.array(img2,dtype= np.uint8)
        ####### load labels ############
        if self.flag == 'train' or self.flag == 'val':

            label = Image.open(label_path)
            roi = Image.open(roi_path)
            roi = roi.convert("L")
            ###############
            # edge_label = cv2.imread(label_path, 0)
            # edge_label = util.extract_edge_label(edge_label)
            ###############

            if self.transform_med != None:
                label = self.transform_med(label)
                roi = self.transform_med(roi)
                ########################
                # edge_label = self.transform_med(edge_label)
            #print("np shape before: ", np.shape(label))
            label = np.array(label,dtype=np.int32)[:,:,np.newaxis].astype("int")
            #label = np.array(label, dtype=np.int32)[:, :, 0][:,:,np.newaxis].astype("int")
            #print("np shape after: ", np.shape(label))

            roi = np.array(roi, dtype=np.int32) / 255
        else:
            label = np.zeros((height,width,3),dtype=np.uint8)
            edge_label = np.zeros((height, width, 3), dtype=np.uint8)
        if self.transform:
            input_, mask_, roi_ = self.data_transform(img1, img2, label, roi)

        '''''''''
        if self.flag == 'val':
            label = Image.open(label_path)
            if self.transform_med != None:
               label = self.transform_med(label)
            label = np.array(label,dtype=np.int32)
        '''''''''
        #return input_, mask_, roi_
        return input_, mask_

    def __len__(self):

        return len(self.img_label_path_pairs)



class Dataset_val(Dataset):

    def __init__(self,img_path,label_path,file_name_txt_path,split_flag, transform=True, transform_med = None):

        self.label_path = label_path
        self.img_path = img_path
        self.img_txt_path = file_name_txt_path
        self.imgs_path_list = np.loadtxt(self.img_txt_path,dtype=str)
        self.flag = split_flag
        self.transform = transform
        self.transform_med = transform_med
        self.img_label_path_pairs = self.get_img_label_path_pairs()

    def get_img_label_path_pairs(self):
        img_label_pair_list = {}

        if self.flag == 'val':
            self.label_ext = '.png'
            for idx, did in enumerate(open(self.img_txt_path)):
                try:
                    image1_name, image2_name, mask_name, roi_name = did.strip("\n").split(' ')
                except ValueError:
                    image_name = mask_name = did.strip("\n")
                extract_name = image1_name[image1_name.rindex('/') + 1: image1_name.rindex('.')]

                img1_file = os.path.join(self.img_path, image1_name)
                img2_file = os.path.join(self.img_path, image2_name)
                lbl_file = os.path.join(self.label_path, mask_name)
                roi_file = os.path.join(self.img_path, roi_name)
                img_label_pair_list.setdefault(idx, [img1_file, img2_file, lbl_file, roi_file, image2_name])

        if self.flag == 'test':

            for idx, did in enumerate(open(self.img_txt_path)):
                image1_name, image2_name = did.strip("\n").split(' ')
                img1_file = os.path.join(self.img_path, image1_name)
                img2_file = os.path.join(self.img_path, image2_name)
                img_label_pair_list.setdefault(idx, [img1_file, img2_file, None, None, image2_name])

        return img_label_pair_list

    def data_transform(self, img1,img2,lbl, roi):
        img1 = img1[:, :, ::-1]  # RGB -> BGR
        img1 = img1.astype(np.float32)
        img1 -= T0_MEAN_VALUE
        img1 = img1.transpose(2, 0, 1) # H×W×C -> C×H×W
        img1 = img1/128.0 - 1.0

        #img1 = torch.from_numpy(img1).float()

        img2 = img2[:, :, ::-1]  # RGB -> BGR
        img2 = img2.astype(np.float32)
        img2 -= T1_MEAN_VALUE
        img2 = img2.transpose(2, 0, 1)
        #print("unique img2: ", np.unique(img2))
        img2 = img2 / 128.0 - 1.0

        #img2 = torch.from_numpy(img2).float()
        crop_width = 256
        _, h, w = img1.shape
        #print("after h, w: ", img1.shape)

        lbl = lbl.transpose(2, 0, 1)

        mask_ = torch.from_numpy(lbl).long()
        roi_ = torch.from_numpy(roi).long()


        return img1, img2, mask_, h, w

    def __getitem__(self, index):

        img1_path,img2_path,label_path,roi_path, filename = self.img_label_path_pairs[index]
        ####### load images #############
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        height,width,_ = np.array(img1,dtype= np.uint8).shape

        print("height, width: ", height,width)

        if self.transform_med != None:
           img1 = self.transform_med(img1)
           img2 = self.transform_med(img2)
        img1 = np.array(img1,dtype= np.uint8)
        img2 = np.array(img2,dtype= np.uint8)
        if self.flag == 'train' or self.flag == 'val':

            label = Image.open(label_path)
            roi = Image.open(roi_path)
            roi = roi.convert("L")

            if self.transform_med != None:
                label = self.transform_med(label)
                roi = self.transform_med(roi)
            label = np.array(label,dtype=np.int32)[:,:,np.newaxis].astype("int")
            #label = np.array(label, dtype=np.int32)[:, :, 0][:, :, np.newaxis].astype("int")
            roi = np.array(roi, dtype=np.int32) / 255
        else:
            label = np.zeros((height,width,3),dtype=np.uint8)
            edge_label = np.zeros((height, width, 3), dtype=np.uint8)
        if self.transform:
            img1, img2, mask_, h, w = self.data_transform(img1, img2, label, roi)

        return img1, img2, mask_, width,height, h, w

    def __len__(self):

        return len(self.img_label_path_pairs)


