class args:
    PATH = '../data/'
    MODEL_PATH = '../models/'
    bad_files = ["ID_4d238ae90", "ID_1a5a10365", "ID_408f58e9f","ID_bb1d991f6","ID_c44983aeb"]
    # debug mode
    DEBUG = False # debug mode for sanity check

    # original image width and height
    ORIG_W = 3384
    ORIG_H = 2710
    
    IMG_WIDTH = 2048 # training image width
    IMG_HEIGHT = 512  # training image height
    MARGIN_W = ORIG_W // 4
    MODEL_SCALE = 8 # downsample ratio
    FX, FY = 2304.5479,  2305.8757 # camera params
    CX, CY = 1686.2379, 1354.9849 # camera params
    IMG_SHAPE = (2710, 3384, 3)

    # training params
    BATCH_SIZE = 2
    USEMASK = True
    AMP = False # automatic mixed precision training
    MULTI_GPU = False
    DEVICE_ID = 1 # GPU ID
    n_epochs = 50 
    n_worker = 10 # number of cpu threads
    seed = 888 
    DISTANCE_THRESH_CLEAR = 2 # distance post processing threshold
    heatmap_threshold = 0.45    