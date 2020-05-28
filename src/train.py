import os
import gc
import time
import warnings

from tqdm import tqdm
import numpy as np
import torch
import torchvision.utils as torch_utils
import torch.optim as torch_optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms as torch_transforms

from consts import (BATCH_SIZE, LRS_MODE, TT_TRAIN_DIR, TT_TEST_DIR,
                    TT_TRAIN_GT_DIR, TT_TEST_GT_DIR, NO_EPOCHS, LOG_ITER)
from data_loaders import (load_metadata, TotalTextDatasetIter)
from losses import DBLoss
from lr_schedulers import WarmupPolyLR
from models import DBTextModel
from text_metrics import (cal_text_score, runningScore)
from utils import (to_device, minmax_scaler_img)


warnings.filterwarnings('ignore')


def get_data_loaders(batch_size=4):

    # train
    tt_train_img_fps, tt_train_gt_fps = \
        load_metadata(TT_TRAIN_DIR, TT_TRAIN_GT_DIR)
    # test
    tt_test_img_fps, tt_test_gt_fps = \
        load_metadata(TT_TEST_DIR, TT_TEST_GT_DIR)

    totaltext_train_iter = TotalTextDatasetIter(tt_train_img_fps,
                                                tt_train_gt_fps,
                                                debug=False)
    totaltext_test_iter = TotalTextDatasetIter(tt_test_img_fps,
                                               tt_test_gt_fps,
                                               debug=False)

    totaltext_train_loader = DataLoader(dataset=totaltext_train_iter,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=1)
    totaltext_test_loader = DataLoader(dataset=totaltext_test_iter,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=1)
    return totaltext_train_loader, totaltext_test_loader


def main():

    # setup model
    assert os.path.exists("./logs")
    tfb_log_dir = './logs/{}/'.format(int(time.time()))
    print(tfb_log_dir)
    if not os.path.exists(tfb_log_dir):
        os.makedirs(tfb_log_dir)
    tfb_writer = SummaryWriter(tfb_log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dbnet = DBTextModel().to(device)
    dbnet.train()
    criterion = DBLoss(alpha=1, beta=10, negative_ratio=3,
                       reduction='mean').to(device)
    db_optimizer = torch_optim.Adam(dbnet.parameters(),
                                    lr=0.001,
                                    weight_decay=0.0,
                                    amsgrad=False)

    # setup model checkpoint
    best_test_loss = np.inf
    best_train_loss = np.inf

    db_scheduler = None
    if LRS_MODE == 'poly':
        db_scheduler = WarmupPolyLR(db_optimizer, warmup_iters=100)
    elif LRS_MODE == 'reduce':
        db_scheduler = torch_optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=db_optimizer,
            mode='min',
            factor=0.1,
            patience=4,
            verbose=True)

    # get data loaders
    totaltext_train_loader, totaltext_test_loader = get_data_loaders(
        batch_size=BATCH_SIZE)

    # train model
    torch.cuda.empty_cache()
    gc.collect()
    global_steps = 0
    for epoch in range(NO_EPOCHS):

        # TRAINING
        dbnet.train()
        train_loss = 0
        running_metric_text = runningScore(2)
        for batch_index, batch in enumerate(totaltext_train_loader):
            lr = db_optimizer.param_groups[0]['lr']
            global_steps += 1

            # resized_image, prob_map, supervision_mask, threshold_map, text_area_map  # noqa
            batch = to_device(batch, device=device)
            img_fps, imgs, prob_maps, supervision_masks, threshold_maps, text_area_maps = batch  # noqa

            preds = dbnet(imgs)
            assert preds.size(1) == 3

            _batch = torch.stack(
                [prob_maps, supervision_masks, threshold_maps, text_area_maps])
            prob_loss, threshold_loss, binary_loss, total_loss = criterion(
                preds, _batch)
            db_optimizer.zero_grad()

            # prob_loss, threshold_loss, binary_loss, total_loss
            total_loss.backward()
            db_optimizer.step()
            if LRS_MODE == 'poly':
                db_scheduler.step()

            # acc iou: pred_prob_map, gt_prob_map, supervision map, 0.3
            score_shrink_map = cal_text_score(preds[:, 0, :, :],
                                              prob_maps,
                                              supervision_masks,
                                              running_metric_text,
                                              thred=0.3)

            train_loss += total_loss
            acc = score_shrink_map['Mean Acc']
            iou_shrink_map = score_shrink_map['Mean IoU']

            # tf-board
            tfb_writer.add_scalar('TRAIN/LOSS/loss', total_loss, global_steps)
            tfb_writer.add_scalar('TRAIN/LOSS/prob_loss', prob_loss,
                                  global_steps)
            tfb_writer.add_scalar('TRAIN/LOSS/threshold_loss', threshold_loss,
                                  global_steps)
            tfb_writer.add_scalar('TRAIN/LOSS/binary_loss', binary_loss,
                                  global_steps)
            tfb_writer.add_scalar('TRAIN/ACC_IOU/acc', acc, global_steps)
            tfb_writer.add_scalar('TRAIN/ACC_IOU/iou_shrink_map',
                                  iou_shrink_map, global_steps)
            tfb_writer.add_scalar('TRAIN/HPs/lr', lr, global_steps)

            if global_steps % LOG_ITER == 0:
                print("[{}-{}] - lr: {} - loss: {} - acc: {} - iou: {}".format(
                    epoch + 1, global_steps, lr,
                    total_loss, acc, iou_shrink_map)
                )

        end_epoch_loss = train_loss / len(totaltext_train_loader)
        print(">>> Train loss: {}".format(end_epoch_loss))
        gc.collect()

        # TFB IMGs
        prob_threshold = 0.5

        # origin img
        # imgs.shape = (batch_size, 3, image_size, image_size)
        imgs = torch.stack([
            torch.Tensor(
                minmax_scaler_img(img_.to('cpu').numpy().transpose((1, 2, 0))))
            for img_ in imgs
        ])
        imgs = torch.Tensor(imgs.numpy().transpose((0, 3, 1, 2)))
        imgs_grid = torch_utils.make_grid(imgs)
        imgs_grid = torch.unsqueeze(imgs_grid, 0)
        # imgs_grid.shape = (3, image_size, image_size * batch_size)
        tfb_writer.add_images('TRAIN/origin_imgs', imgs_grid, global_steps)

        # pred_prob_map / pred_thresh_map
        pred_prob_map = preds[:, 0, :, :]
        pred_threshold_map = preds[:, 1, :, :]
        pred_prob_map[pred_prob_map <= prob_threshold] = 0
        pred_prob_map[pred_prob_map > prob_threshold] = 1

        # make grid
        pred_prob_map = pred_prob_map.unsqueeze(1)
        pred_threshold_map = pred_threshold_map.unsqueeze(1)

        probs_grid = torch_utils.make_grid(pred_prob_map, padding=0)
        probs_grid = torch.unsqueeze(probs_grid, 0)
        probs_grid = probs_grid.detach().to('cpu')

        thres_grid = torch_utils.make_grid(pred_threshold_map, padding=0)
        thres_grid = torch.unsqueeze(thres_grid, 0)
        thres_grid = thres_grid.detach().to('cpu')

        tfb_writer.add_images('TRAIN/prob_imgs', probs_grid, global_steps)
        tfb_writer.add_images('TRAIN/thres_imgs', thres_grid, global_steps)

        # EVAL
        dbnet.eval()
        test_loss = 0
        for val_batch_index, test_batch in tqdm(
                enumerate(totaltext_test_loader),
                total=len(totaltext_test_loader)):

            with torch.no_grad():
                test_batch = to_device(test_batch, device=device)
                img_fps, imgs, prob_maps, supervision_masks, threshold_maps, text_area_maps = test_batch  # noqa

                test_preds = dbnet(imgs)
                assert test_preds.size(1) == 2

                _batch = torch.stack([
                    prob_maps, supervision_masks, threshold_maps,
                    text_area_maps
                ])
                test_total_loss = criterion(test_preds, _batch)
                test_loss += test_total_loss

                test_score_shrink_map = cal_text_score(test_preds[:, 0, :, :],
                                                       prob_maps,
                                                       supervision_masks,
                                                       running_metric_text,
                                                       thred=0.3)
                test_acc = test_score_shrink_map['Mean Acc']
                test_iou_shrink_map = test_score_shrink_map['Mean IoU']
                tfb_writer.add_scalar('TEST/LOSS/val_loss', test_total_loss,
                                      global_steps)
                tfb_writer.add_scalar('TEST/ACC_IOU/val_acc', test_acc,
                                      global_steps)
                tfb_writer.add_scalar('TEST/ACC_IOU/val_iou_shrink_map',
                                      test_iou_shrink_map, global_steps)

        test_loss = test_loss / len(totaltext_test_loader)
        print("[{}] - test_loss: {}".format(global_steps, test_loss))

        if test_loss <= best_test_loss and train_loss < best_train_loss:
            best_test_loss = test_loss
            best_train_loss = train_loss
            torch.save(
                dbnet.state_dict(),
                "./models/best_cps.pth"
            )

        if LRS_MODE == 'reduce':
            db_scheduler.step(test_loss)
        torch.cuda.empty_cache()
        gc.collect()

    torch.save(dbnet.state_dict(), "./models/last_cps.pth")


if __name__ == '__main__':
    main()
