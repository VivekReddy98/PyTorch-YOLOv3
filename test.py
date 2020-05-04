#!/bin/python3
'''
sjoshi26 shashank joshi
akwatra archit kwatra
vkarri vivek reddy karri
'''

from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

# Exactly the same as utils.ListDataset() class but it takes the input as a list than a path.
class ListDatasetCustom(Dataset):
    def __init__(self, img_files, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        self.img_files = img_files

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)

# Fucntion to split the work among ranks
def splitData(list_path, rank, size):
    with open(list_path, "r") as file:
        img_files = file.readlines()

    numFiles = len(img_files)
    numfilesPRank = int(numFiles/size)

    # Find Low and Max indices
    low = rank*numfilesPRank
    max = rank*numfilesPRank+numfilesPRank

    if(rank == size-1):
        max = max + numFiles%size

    # Return those image files
    return img_files[low:max]

def evaluate(model, img_files, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    total_inference_time = 0
    total_data_loading_time = 0

    start_data_load = time.time()
    # Get dataloader
    dataset = ListDatasetCustom(img_files, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)

    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            start = time.time()
            outputs = model(imgs)
            end = time.time()
            total_inference_time += end-start
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    end_data_load = time.time()

    total_data_loading_time = end_data_load - start_data_load - total_inference_time

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]

    # precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return true_positives, pred_scores, pred_labels, labels, total_inference_time, total_data_loading_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)


    # ------------------- Code to set up the distributed Environment ---------------
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    ip_addr = os.environ['MASTER_ADDR']
    size = int(os.environ['SLURM_NTASKS'])
    print("rank : %d" %rank)
    print("size : %d" %size)
    backend = 'nccl'
    method = 'tcp://' + ip_addr + ":22233"
    torch.distributed.init_process_group(backend, world_size=size, rank=rank, init_method=method)
    # ------------------------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Split the datafiles list according to the rank
    myfilenames = splitData(valid_path, rank, size)

    total_testing_time_start = time.time()

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    # Get the Local Results out
    true_positives_local, pred_scores_local, pred_labels_local, labels_local, total_inference_time, total_data_loading_time = evaluate(
        model,
        img_files=myfilenames,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
    )

    total_testing_time_end = time.time()

    true_positives_local = true_positives_local.reshape((-1,1))
    pred_scores_local = pred_scores_local.reshape((-1,1))
    pred_labels_local = pred_labels_local.reshape((-1,1))


    # Reshape, Concatenate and Cast Local Variables for Broadcasting.
    allMetricMatrix = np.concatenate((true_positives_local, pred_scores_local, pred_labels_local), axis=1)
    allMetricMatrix = allMetricMatrix.astype('float32')


    # -------------  Communicate all the locally found tensors to rank 0 ---------------------------------------
    if (rank == 1):
        # Broascast Sizes of the Tensors, just so that rank 1 can allocate necessary sizes
        size_tensors = torch.FloatTensor([allMetricMatrix.shape[0], allMetricMatrix.shape[1], len(labels_local)]).cuda(device)
        torch.distributed.broadcast(size_tensors, rank)

        # Send All Metrics Matrix
        allMetrics_send = torch.from_numpy(allMetricMatrix).cuda(device)
        torch.distributed.broadcast(allMetrics_send, rank)

        # Send labels List
        labels_send = torch.FloatTensor(labels_local).cuda(device)
        torch.distributed.broadcast(labels_send, rank)

    else:
        # Recieve the Sizes of tensors to be recieved.
        recv_size_tensor = torch.FloatTensor([0., 0., 0.]).cuda(device)
        torch.distributed.reduce(recv_size_tensor, 0)
        recv_size_list = recv_size_tensor.tolist()
        recv_size_list = [int(i) for i in recv_size_list]

        # Recieve All Metrics Matrix
        allMetrics_recv = torch.zeros([recv_size_list[0], recv_size_list[1]], dtype=torch.float).cuda(device)
        torch.distributed.reduce(allMetrics_recv, 0)
        allMetricMatrixNew = allMetrics_recv.cpu().numpy()

        # Recieve labels List
        labels_recv = torch.zeros([recv_size_list[2]], dtype=torch.float).cuda(device)
        torch.distributed.reduce(labels_recv, 0)
        labels_new = labels_recv.tolist()

        # Concatenate Local and Global Results
        allMatrix = np.concatenate((allMetricMatrix, allMetricMatrixNew), axis=0)
        labels = labels_local + labels_new

        # Find AP and mAP
        precision, recall, AP, f1, ap_class = ap_per_class(allMatrix[:,0], allMatrix[:,1], allMatrix[:,2], labels)

        # Print the results
        print("Average Precisions:")
        for i, c in enumerate(ap_class):
            print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

        print(f"mAP: {AP.mean()}")
    # -------------  Communicate all the locally found tensors to rank 0 ---------------------------------------

    # Barrier
    torch.distributed.barrier()

    # Print the Time taken by both the ranks
    print("------------------Rank : {}----------------------".format(rank))
    print("Total Data Loading Time Taken {} secs".format(total_data_loading_time))
    print("Total Inference Time Taken: {0} secs".format(total_inference_time))
