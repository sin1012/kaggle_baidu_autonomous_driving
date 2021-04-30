from utils import *
from conf import *
from transforms import get_train_transforms
from dataset import CompetitionDataset
from models import *
import warnings 
warnings.filterwarnings('ignore')

train = pd.read_csv(args.PATH + 'train.csv')
test = pd.read_csv(args.PATH + 'sample_submission.csv')

seed_everything(args.seed)
# remove bad files
train = train.set_index("ImageId").drop(index = args.bad_files).reset_index()
print('[🐻] Bad Image files removed.')

# debug mode
if args.DEBUG:
    print('[✔️] Debug Mode...')
    train = train[:10]
    test  = test[:5]
else:
    print('[✔️] Full Training Mode...')

print('[🐶] Building Dataset.')
train_images_dir = args.PATH + 'train_images/{}.jpg'
test_images_dir = args.PATH + 'test_images/{}.jpg'
train_masks_dir = args.PATH + 'train_masks/{}.jpg'
test_masks_dir = args.PATH + 'test_masks/{}.jpg'

df_train, df_valid = train_test_split(train, test_size=0.1, random_state=args.seed)
df_test = test

train_dataset = CompetitionDataset(df_train, train_images_dir, train_masks_dir, mode='train', transforms=get_train_transforms())
valid_dataset = CompetitionDataset(df_valid, train_images_dir, train_masks_dir, mode='valid')
test_dataset = CompetitionDataset(df_test, test_images_dir, test_masks_dir, mode='test')

train_loader = DataLoader(dataset=train_dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers = args.n_worker)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.BATCH_SIZE, shuffle=False, num_workers = args.n_worker)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.BATCH_SIZE, shuffle=False, num_workers = args.n_worker)

device = torch.device(f"cuda:{args.DEVICE_ID}" if torch.cuda.is_available() else "cpu")

print(f'[💻] Training on GPU {args.DEVICE_ID}')

model = CentResnet(8).to(device)
if args.MULTI_GPU:
    model = nn.DataParallel(model)
print('[✔️] Model Loaded.')

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True, min_lr=1e-3*1e-3, factor=1./9.)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs, eta_min=1e-5)

print('[🚀] Start Training...')
if args.AMP:
    print('[🍔] Automatic Mixed Precision Training.')

def mask_forward(img, ignore_mask):  
    '''
    [✔️] use mask as another channel and feed it into the model
    input:
        img: image
        ignore_mask: mask for non-interested images
    '''
    return model(torch.from_numpy(np.concatenate((img, ignore_mask),axis=1)).to(device) )

def train_func(epoch, history=None):
    model.train()
    print('Train Epoch: {} \tLR: {:.6f}'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    bar = tqdm(train_loader)
    if args.AMP:
        scaler = torch.cuda.amp.GradScaler()

    for batch_idx, (img_batch, mask_batch, regr_batch, ignore_mask_batch, ignore_mask_for_feed_batch) in enumerate(bar):
        mask_batch = mask_batch.to(device)
        regr_batch = regr_batch.to(device)
        if args.AMP:
            with torch.cuda.amp.autocast(): # todo: fix amp stability issue
                if args.USEMASK:
                    output = mask_forward(img_batch, ignore_mask_for_feed_batch)
                else:
                    output = model(img_batch)
                loss, mask_loss, regr_loss = criterion(output, mask_batch, regr_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            if args.USEMASK:
                output = mask_forward(img_batch, ignore_mask_for_feed_batch)
            else:
                output = model(img_batch)
            loss, mask_loss, regr_loss = criterion(output, mask_batch, regr_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        bar.set_description(f'Sum Loss: {loss:.3f}, Keypoint loss: {mask_loss:.2f}, Regression loss: {regr_loss:.4f}')
        
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
            history.loc[epoch + batch_idx / len(train_loader), 'train_keypoint_loss'] = mask_loss.data.cpu().numpy()
            history.loc[epoch + batch_idx / len(train_loader), 'train_regr_loss'] = regr_loss.data.cpu().numpy()

    
    print('Train Loss: {:.4f}\tKeypoint: {:.4f}\tRegression: {:.4f}'.format(
        history.query(f"index >= {epoch}").train_loss.mean(),
        history.query(f"index >= {epoch}").train_mask_loss.mean(),
        history.query(f"index >= {epoch}").train_regr_loss.mean() ))

def valid_func(epoch, history=None):
    model.eval()
    loss = 0
    valid_loss = 0
    valid_mask_loss = 0
    valid_regr_loss = 0
    valid_predictions = []  

    with torch.no_grad():
        for img_batch, mask_batch, regr_batch, ignore_mask_batch, ignore_mask_for_feed_batch in valid_loader:

            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)

            output = mask_forward(img_batch, ignore_mask_for_feed_batch)

            loss,mask_loss, regr_loss = criterion(output, mask_batch, regr_batch)
            valid_loss += loss.data
            valid_mask_loss += mask_loss.data
            valid_regr_loss += regr_loss.data

            for out, ignore_mask in zip(output.data.cpu().numpy(), ignore_mask_batch):  
                coords = extract_coords(out, ignore_mask)
                s = coords2str(coords)
                valid_predictions.append(s)  

    valid_loss /= len(valid_loader)
    valid_mask_loss /= len(valid_loader)
    valid_regr_loss /= len(valid_loader)
    
    if history is not None:
        history.loc[epoch, 'valid_loss'] = valid_loss.cpu().numpy()
        history.loc[epoch, 'keypoint_loss'] = valid_mask_loss.cpu().numpy()
        history.loc[epoch, 'regr_loss'] = valid_regr_loss.cpu().numpy()

    CV_valid = pd.DataFrame()
    CV_valid["ImageId"] = df_valid.ImageId.copy()
    CV_valid['PredictionString'] = valid_predictions
    mapval = calc_map(CV_valid)
    history.loc[epoch, 'map'] = mapval

    print('Valid Loss: {:.4f}\tKeypoint: {:.4f}\tRegression: {:.4f}\tmAP: {:.6f}'.format(
            valid_loss, valid_mask_loss, valid_regr_loss, mapval))

import gc

history = pd.DataFrame()

os.makedirs(args.MODEL_PATH, exist_ok=True)
for epoch in range(args.n_epochs):
    torch.cuda.empty_cache()
    gc.collect()
    train_func(epoch, history)
    valid_func(epoch, history)
    if epoch >= 10:
        if args.MULTI_GPU:
            torch.save(model.module.state_dict(), f'DP_{args.MODEL_PATH}model_epoch_{epoch}.pth')
        else:
            torch.save(model.state_dict(), f'{args.MODEL_PATH}model_epoch_{epoch}.pth')
    scheduler.step()

history.to_csv(f'{args.MODEL_PATH}training_history.csv', index=False)
print(f'History successfully saved at {args.MODEL_PATH}')