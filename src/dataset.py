from utils import *
from conf import *

class CompetitionDataset(Dataset):
    '''
    Competition Dataset:
    Initialization:
        dataframe: original csv
        root_dir: directory of images
        mask_root_dir: directory of mask
        mode: mode for dataset creation, train, valid or test
        transforms: custom training transformation
    Output:
        img: preprocessed images
        hm: heatmap
        regr: regression targets
        ignore_img: original mask for car out of interest
        ignore_img_forward: mask for forward, same aspect ratio as preprocessed images
    '''

    def __init__(self, dataframe, root_dir, mask_root_dir, mode='train', transforms=False):
        self.df = dataframe # dataframe, csv
        self.root_dir = root_dir # root directory for images
        self.mask_root_dir = mask_root_dir # root directory for images
        self.mode = mode # mode for training or inference 
        self.transforms = transforms # custom transformation
        print(f'[✔️] Dataset initiated in {mode} mode.')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image name
        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)
        
        img = cv2.imread(img_name)[:,:,::-1] # bgr to rgb

        if self.transforms: 
            img = self.transforms(image=img)["image"] # apply transformation

        if self.mode == 'train': # if training mode, perform flip augmentation with p = 0.5
            flip = False
            if np.random.rand() < 0.5: 
                flip = True # hflip
        elif self.mode == 'valid' or self.mode == 'test': # no flip during validation or test
            flip = False 

        img = preprocess_image(img, flip=flip) # preprocessing
        img = np.rollaxis(img, 2, 0) # channel first
        
        # the dataset also contains mask the masks for cars that are not of interest
        ignore_img = cv2.imread(self.mask_root_dir.format(idx), cv2.IMREAD_GRAYSCALE)
        if ignore_img is None:  # some pics has no mask, we create fake mask with all zeros
            ignore_img = np.zeros((args.ORIG_H, args.ORIG_W), dtype='float32')

        ignore_img = np.array(ignore_img).astype('float32') / 255.
        # do this for mask to get the same aspect ratio 
        ignore_img_forward = preprocess_mask_image(ignore_img)
        #  h,w -> 1,h,w, add a channel(gray)
        ignore_img_forward = np.expand_dims(ignore_img_forward, 0) 
        
        # Get heatmap and regression maps
        if self.mode != 'test': 
            hm, regr = get_hm_and_regr(img, labels, flip)
            regr = np.rollaxis(regr, 2, 0) # channel first
        else:
            hm, regr = 0, 0 # no heatmap and regression target if we are at test phase
        
        return [img, hm, regr, ignore_img, ignore_img_forward]