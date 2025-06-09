from __future__ import absolute_import, division, print_function
import os
import argparse
import torch
import warnings

from trainer_sam4em import Trainer_sam4em

warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('Baseline training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='START of CHECKPOINT',
                        help='Checkpoint and resume ')

    # Model parameters
    parser.add_argument('--model_name', default='hsam', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--img_size', default=512,
                        type=int, help='images input size')
    parser.add_argument('--prompt_type', default="",
                        type=str, help='Prompt type e.g points, boxes, and masks')

    # Dataset parameters
    parser.add_argument('--dataset', default='lucchi', type=str)
    parser.add_argument('--root_path', default='../datasets/Lucchi', type=str, metavar='DATASET PATH',
                        help='Path to dataset')
    parser.add_argument('--input_dir', type=str,
                        default='img', help='list inputs image dir')
    parser.add_argument('--target_dir', type=str,
                        default='label', help='list target image dir')
    parser.add_argument('--train_folders', type=str, nargs='+', help='list of training folders')
    parser.add_argument('--test_folders', type=str, nargs='+', help='list of testing folders')
    parser.add_argument('--val_folders', type=str, nargs='+', help='list of testing folders')
    parser.add_argument('--num_classes', type=int,
                        default=1, help='output channel of network')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.0005,
                        help='segmentation network learning rate')
    parser.add_argument('--vit_name', type=str,
                        default='vit_b', help='select one vit model')
    parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--load_weights_dir', type=str, default=None,
                        help='Pretrained model checkpoint for evaluation')
    parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(__file__), "tmp"), help="log directory")
    parser.add_argument("--log_frequency", type=int, default=100, help="number of batches between each tensorboard log")
    parser.add_argument('--warmup', action='store_true', default=True, help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--warmup_period', type=int, default=250,
                        help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('--lora', default=100, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--double_mask', default=True, type=bool,
                        help='Show some outputs')
    parser.add_argument('--full_tuning', default=False, type=bool,
                        help='Show some outputs')
    parser.add_argument('--adapter', default=False, type=bool,
                        help='Show some outputs')
    parser.add_argument('--embedd', default=False, type=bool,
                        help='Show some outputs')
    parser.add_argument('--mixup', default=False, type=bool,
                        help='Show some outputs')
    parser.add_argument('--use_two_mask', default=False, type=bool,
                        help='Show some outputs')
    parser.add_argument('--img_encoder', default="sam2", type=str,
                        help='iamge encoder sam1 or sam2')
    parser.add_argument('--mem_attention', default=False, type=bool,
                        help='mem attention either true or false')
    parser.add_argument('--threshold', default=0.4, type=float)
    parser.add_argument('--mem_slot', default=8, type=int)
    parser.add_argument('--dice_parm', default=0.5, type=float)
    parser.add_argument('--mask1_parm', default=0.5, type=float)
    parser.add_argument('--mem_avg_parm', default=0.3, type=float)
    parser.add_argument('--mask_value', default=0, type=int)
    return parser

def main(args):
    # Set the GPU device
    torch.cuda.set_device(args.local_rank)
    trainer = Trainer_sam4em(args)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'SAM4EM training and evaluation script for EM dataset', parents=[get_args_parser()])

    args = parser.parse_args()
    main(args)
