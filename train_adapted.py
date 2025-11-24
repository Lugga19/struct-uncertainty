from dataclasses import dataclass

import torch
import numpy as np
import argparse, json
import os, glob, sys, shutil
import wandb
from time import time

from dataloader import DRIVE
from datasets import TrainDataset, TestDataset
from unet.unet_model import UNet
from edgedetector.edgedetector import LDC

from dmt_trainer import getData_train, getData_val
from unc_model import UncertaintyModel, UncertaintyModel_GNN
from utilities import MSE_VAR


class Args():
    def __init__(self, list_name):
        self.train_list = list_name
        self.train_data = "SEM"  # add crop_img true


def parse_func(args):
    ### Reading the parameters json file
    print("Reading params file {}...".format(args.params))
    with open(args.params, 'r') as f:
        params = json.load(f)

    activity = params['common']['activity']
    mydict = {}
    mydict['num_classes'] = int(params['common']['num_classes'])
    mydict['folders'] = [params['common']['img_folder'], params['common']['gt_folder']]
    mydict["segmodel_checkpoint_restore"] = params['common']['segmodel_checkpoint_restore']
    mydict['dataname'] = params['common']['dataname']
    mydict['network'] = params['common']['network'].lower()
    mydict['training_folder'] = params['common']['training_folder']
    mydict['val_folder'] = params['common']['val_folder']

    if activity == "train":
        mydict['train_datalist'] = params['train']['train_datalist']
        mydict['validation_datalist'] = params['train']['validation_datalist']
        mydict['output_folder'] = params['train']['output_folder']
        mydict['learning_rate'] = float(params['train']['learning_rate'])

        mydict['num_epochs'] = int(params['train']['num_epochs']) + 1
        mydict['save_every'] = params['train']['save_every']
        # Optional early stopping parameters
        # If 'early_stopping_patience' > 0, early stopping is enabled
        mydict['early_stopping_patience'] = int(params['train'].get('early_stopping_patience', 0))
        mydict['early_stopping_min_delta'] = float(params['train'].get('early_stopping_min_delta', 0.0))

    else:
        print("Wrong activity chosen")
        sys.exit()

    print(activity, mydict)
    return activity, mydict


def set_seed(): # reproductibility 
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def train_func_2d(mydict):
    # Reproducibility, and Cuda setup
    set_seed()
    device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')

    print("CUDA device: {}".format(device))

    if not torch.cuda.is_available():
        print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

    force_cudnn_initialization()

    if not os.path.exists(mydict['output_folder']):
        os.makedirs(mydict['output_folder'])

    if not os.path.exists(os.path.join(mydict['output_folder'],'inputs')):
        shutil.copytree('inputs/', os.path.join(mydict['output_folder'],'inputs'))
    
    if not os.path.exists(os.path.join(mydict['output_folder'],'output')):
        os.makedirs(os.path.join(mydict['output_folder'], 'output'))

    # Train and Val Data       
    if mydict['dataname'].lower() == "drive":
        training_set = DRIVE(mydict['train_datalist'], mydict['folders'], "train")
        validation_set = DRIVE(mydict['validation_datalist'], mydict['folders'], "val") # full image takes too long
        n_channels = 3
        in_channels = 5
    elif mydict['dataname'].lower() == "sem":
        mean_bgr_train = [97.38495689751913] * 3

        dataset_train = TrainDataset(mydict['training_folder'],
                                     img_width=128,
                                     img_height=128,
                                     mean_bgr=mean_bgr_train,
                                     train_mode='train',
                                     arg=Args("training.lst")
                                     )
        mean_bgr_test = [96.22038432767576] * 3
        dataset_val = TrainDataset(mydict['val_folder'],
                                  img_width=128,
                                  img_height=128,
                                  mean_bgr=mean_bgr_test,
                                    train_mode='test',
                                  arg=Args("validation.lst"),
                                  )
        # training_set = SEM(mydict['train_datalist'], mydict['training_folder'], "train")
        # validation_set = SEM(mydict['validation_datalist'], mydict['val_folders'], "val")  # full image takes too long

    print("data loaded")

    training_generator = torch.utils.data.DataLoader(dataset_train,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True)
    validation_generator = torch.utils.data.DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=2)

    # training_generator = torch.utils.data.DataLoader(training_set,batch_size=1,shuffle=True,num_workers=2, drop_last=True)
    # validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=1,shuffle=False,num_workers=2, drop_last=False)
    n_channels = 3
    in_channels = 5
    # Networks : The Seg Model of whose uncertainty we want to predict
    if mydict['network'] == "unet":
        feature_extractor = UNet(n_channels=n_channels, n_classes=mydict['num_classes'], start_filters=64).to(device)
    elif mydict['network'] == "edgedetector":
        feature_extractor = LDC().to(device)

    #unc_regressor = UncertaintyModel_GNN(in_channels=in_channels, num_features=36, hidden_units=48).float().to(device)
    unc_regressor = UncertaintyModel(in_channels=in_channels, num_features=36, hidden_units=48).float().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(unc_regressor.parameters(), lr=mydict['learning_rate'], weight_decay=0)

    # Load checkpoint
    if mydict['segmodel_checkpoint_restore'] != "":
        feature_extractor.load_state_dict(torch.load(mydict['segmodel_checkpoint_restore']), strict=True)
        print("loaded checkpoint! {}".format(mydict['segmodel_checkpoint_restore']))

        for param in feature_extractor.parameters():
            param.requires_grad = False

        print("Freezing weights of Seg Model")
    else:
        print("No checkpoint provided! Aborting...")
        sys.exit()

    # Losses
    unc_loss_func = MSE_VAR()
    
    # Train loop
    best_dict = {}
    best_dict['epoch'] = 0
    best_dict['val_loss'] = None
    # Early stopping state
    patience = mydict.get('early_stopping_patience', 7)
    min_delta = mydict.get('early_stopping_min_delta', 0.0001)
    epochs_no_improve = 0
    use_early_stopping = patience > 0
    print("Let the training begin!")
    wandb.init(project="structural-uncertainty", config=mydict)

    for epoch in range(mydict['num_epochs']):
        torch.cuda.empty_cache() # cleanup
        unc_regressor.to(device).train()

        avg_train_loss = 0.0
        cntvar = 0
        epoch_start_time = time()

        for step, (patch, mask) in enumerate(training_generator):
            optimizer.zero_grad()

            patch = patch.to(device)
            mask = torch.squeeze(mask.to(device),dim=1)

            y_patchlikelihood = feature_extractor(patch)[-1]

            imgbatch, unc_input, unc_gt = getData_train(mydict['num_classes'], mydict['output_folder'], patch, y_patchlikelihood, mask)

            if unc_input is not None:
                imgbatch = imgbatch.float().to(device)
                unc_input = unc_input.float().to(device)
                unc_gt = unc_gt.float().to(device)

                unc_pred_mu, unc_pred_sigma = unc_regressor(imgbatch,unc_input)
                unc_pred_mu = torch.squeeze(unc_pred_mu, dim=1)
                unc_pred_sigma = torch.squeeze(unc_pred_sigma, dim=1)
                
                loss_val = unc_loss_func(unc_pred_mu, unc_pred_sigma, unc_gt)
                avg_train_loss += loss_val
                cntvar+=1

                loss_val.backward()
                optimizer.step()

        if cntvar!=0:
            avg_train_loss /= cntvar

        epoch_end_time = time()
        print("Epoch {} took {} seconds.\nAverage training loss: {}\nNumber of samples in this epoch {}".format(epoch, epoch_end_time-epoch_start_time, avg_train_loss, cntvar))

        validation_start_time = time()
        with torch.no_grad():
            #unc_regressor.eval() # we want to keep dropout
            validation_iterator = iter(validation_generator)
            avg_val_loss = 0.0
            cntvar = 0
            for _ in range(len(validation_generator)):
                x, y_gt = next(validation_iterator)
                x = x.to(device, non_blocking=True)
                y_gt = y_gt.to(device, non_blocking=True)

                y_patchlikelihood = feature_extractor(x)[-1]

                imgbatch, unc_input, unc_gt = getData_val(mydict['num_classes'], mydict['output_folder'], x, y_patchlikelihood, y_gt)

                if unc_input is not None:
                    imgbatch = imgbatch.float().to(device)
                    unc_input = unc_input.float().to(device)
                    unc_gt = unc_gt.float().to(device)

                    unc_pred_mu, unc_pred_sigma = unc_regressor(imgbatch,unc_input)
                    unc_pred_mu = torch.squeeze(unc_pred_mu, dim=1)
                    unc_pred_sigma = torch.squeeze(unc_pred_sigma, dim=1)
                    
                    avg_val_loss -= unc_loss_func(unc_pred_mu, unc_pred_sigma, unc_gt)
                    #negative because we maximize val_loss while MSE_VAR is a minimization objective
                    cntvar+=1

            if cntvar!=0:
                avg_val_loss /= cntvar

        validation_end_time = time()
        print("End of epoch validation took {} seconds.\nAverage validation loss: {}\nNumber of samples in this val-loop: {}\nLearning rate: {}".format(validation_end_time - validation_start_time, avg_val_loss, cntvar, optimizer.param_groups[0]['lr']))

        wandb.log({"Epoch": epoch, "Train Loss": avg_train_loss, "Validation Loss": avg_val_loss, "Learning Rate": optimizer.param_groups[0]['lr']})

        # Check for best epoch and update early stopping counters
        if epoch == 0:
            best_dict['epoch'] = epoch
            best_dict['val_loss'] = avg_val_loss
            # save initial best model
            torch.save(unc_regressor.state_dict(), os.path.join(mydict['output_folder'], "uncertainty_model_best.pth"))
        else:
            # improvement check (note: larger avg_val_loss is better in this codebase)
            if avg_val_loss > best_dict['val_loss'] + min_delta:
                best_dict['val_loss'] = avg_val_loss
                best_dict['epoch'] = epoch
                epochs_no_improve = 0
                torch.save(unc_regressor.state_dict(), os.path.join(mydict['output_folder'], "uncertainty_model_best.pth"))
                artifact = wandb.Artifact(
                    name=f"dmt-uncertainty-model-epoch-{epoch}",
                    type="model",
                )
            else:
                epochs_no_improve += 1

        print("Best epoch so far: {}\n".format(best_dict))

        # Early stopping: stop if no improvement for 'patience' epochs
        if use_early_stopping and epochs_no_improve >= patience:
            print(f"Early stopping: no improvement for {epochs_no_improve} epochs (patience={patience}). Stopping training.")
            wandb.log({"EarlyStopping/StoppedEpoch": epoch, "EarlyStopping/Patience": patience, "EarlyStopping/EpochsNoImprove": epochs_no_improve})
            break
        # save checkpoint for save_every
        if epoch % mydict['save_every'] == 0:
            torch.save(unc_regressor.state_dict(), os.path.join(mydict['output_folder'], "uncertainty_model_epoch" + str(epoch) + ".pth"))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help="Path to the parameters file")
    
    if len(sys.argv) == 1:
        print("Path to parameters file not provided. Exiting...")

    else:
        args = parser.parse_args()
        activity, mydict = parse_func(args)

    if activity == "train":
        wandb.login()
        train_func_2d(mydict)
