#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys

import torch
import torch.nn as nn

from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[5].as_posix())


# from chop.passes.module.transforms import quantize_module_transform_pass
from chop.passes.module.transforms import optical_module_transform_pass
from chop.passes.module import report_trainable_parameters_analysis_pass

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from train_mnist_cnn import test, train, Net

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def load_my_model(model_path, device="cpu"):
    # Load the model from the .pt file
    loaded_model = torch.load(model_path, map_location=device)
    # Set it to evaluation mode (important if it contains layers like BatchNorm or Dropout)
    loaded_model.eval()
    return loaded_model


def test_optical_module_transform_pass():
    # model_path = "mase_output/sample_mnist_cnn.pt"
    # mnist_cnn = load_my_model(model_path)
    model = Net()
    # Sanity check and report
    pass_args = {
        "by": "name",
        "fc1": {
            "config": {
                "name": "morr",
                "miniblock": 4,
                "morr_init": True,
                "trainable_morr_bias": False,
                "trainable_morr_scale": False,
            }
        },
        "conv1": {
            "config": {
                "name": "morr",
                "miniblock": 4,
                "morr_init": True,
                "trainable_morr_bias": False,
                "trainable_morr_scale": False,
            }
        },
    }
    optical_module_transform_pass(model, pass_args)
    # torch.save(onn_cnn, "mase_output/onn_cnn.pt")

test_optical_module_transform_pass()










# if __name__ == '__main__':
#     finetune = False

#     if True:
#         parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#         parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                             help='input batch size for training (default: 64)')
#         parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                             help='input batch size for testing (default: 1000)')
#         parser.add_argument('--epochs', type=int, default=14, metavar='N',
#                             help='number of epochs to train (default: 14)')
#         parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
#                             help='learning rate (default: 1.0)')
#         parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
#                             help='Learning rate step gamma (default: 0.7)')
#         parser.add_argument('--no-cuda', action='store_true', default=False,
#                             help='disables CUDA training')
#         parser.add_argument('--no-mps', action='store_true', default=False,
#                             help='disables macOS GPU training')
#         parser.add_argument('--dry-run', action='store_true', default=False,
#                             help='quickly check a single pass')
#         parser.add_argument('--seed', type=int, default=1, metavar='S',
#                             help='random seed (default: 1)')
#         parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                             help='how many batches to wait before logging training status')
#         parser.add_argument('--save-model', action='store_true', default=True,
#                             help='For Saving the current Model')
#         parser.add_argument('--gpu-id', type=int, default=0,
#                     help='Which GPU device to use [default: 0]')

#         args = parser.parse_args()
#         use_cuda = not args.no_cuda and torch.cuda.is_available()
#         use_mps = not args.no_mps and torch.backends.mps.is_available()

#         torch.manual_seed(args.seed)

#         if not args.no_cuda and torch.cuda.is_available():
#             device = torch.device(f"cuda:{args.gpu_id}")
#         elif use_mps:
#             device = torch.device("mps")
#         else:
#             device = torch.device("cpu")

#         train_kwargs = {'batch_size': args.batch_size}
#         test_kwargs = {'batch_size': args.test_batch_size}
#         if use_cuda:
#             cuda_kwargs = {'num_workers': 1,
#                         'pin_memory': True,
#                         'shuffle': True}
#             train_kwargs.update(cuda_kwargs)
#             test_kwargs.update(cuda_kwargs)

#         transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#             ])
#         dataset1 = datasets.MNIST('../data', train=True, download=True,
#                         transform=transform)
#         dataset2 = datasets.MNIST('../data', train=False,
#                         transform=transform)
#         train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
#         test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

#     # load pre-trained cnn
#     cnn = load_my_model("mase_output/sample_mnist_cnn.pt", device)
#     print("-------------- Testing the original cnn model -------------------")
#     _, _ = report_trainable_parameters_analysis_pass(cnn)
#     test(cnn, device, test_loader)

#     ## transform cnn into onn

#     # onn = load_my_model("mase_output/onn_cnn.pt", device)
#     onn_model = perform_optical_module_transform_pass(cnn)
#     onn_model.to(device)
#     print("-------------- Testing the transformed onn model -------------------")
#     _, _ = report_trainable_parameters_analysis_pass(onn_model)
#     test(onn_model, device, test_loader)

#     # Training the onn model
#     if finetune:
#         optimizer = optim.Adadelta(onn_model.parameters(), lr=args.lr)
#         scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
#         for epoch in range(1, args.epochs + 1):
#             train(args, onn_model, device, train_loader, optimizer, epoch)
#             test(onn_model, device, test_loader)
#             scheduler.step()


#         torch.save(onn_model.state_dict(), "mase_output/trained_onn.pt")

#         print("-------------- Testing the trained onn model -------------------")
#         test(onn_model, device, test_loader)
#         _, _ = report_trainable_parameters_analysis_pass(onn_model)


# def perform_optical_module_transform_pass(model, save_path="mase_output/onn_cnn.pt"):
#     # pass_args = {
#     #     "by": "type",
#     #     "linear": {
#     #         "config": {
#     #             "name": "morr",
#     #             "miniblock": 4,
#     #             "morr_init": True,
#     #             "trainable_morr_bias": False,
#     #             "trainable_morr_scale": False,
#     #         }
#     #     },
#     # }
#     pass_args = {
#         "by": "type",
#         "conv2d": {
#             "config": {
#                 "name": "morr",
#                 "miniblock": 4,
#                 "morr_init": True,
#                 "trainable_morr_bias": False,
#                 "trainable_morr_scale": False,
#             }
#         },
#     }
#     onn_model, _ = optical_module_transform_pass(model, pass_args)
#     torch.save(onn_model.state_dict(), save_path)
#     return onn_model
