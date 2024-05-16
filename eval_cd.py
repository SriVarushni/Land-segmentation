from argparse import ArgumentParser
import torch
from models.evaluator import *
import os
import numpy as np
import matplotlib.pyplot as plt

print(torch.cuda.is_available())

"""
eval the CD model
"""

def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='BIT_LEVIR', type=str)
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    parser.add_argument('--output_folder', default='samples/predict', type=str)

    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='quick_start', type=str)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="demo", type=str)
    parser.add_argument('--img_size', default=256, type=int)
    

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='base_transformer_pos_s4_dd8_dedim8', type=str,
                        help='base_resnet18 | base_transformer_pos_s4_dd8 | base_transformer_pos_s4_dd8_dedim8|')
    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)

    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join('checkpoints', args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join('vis', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split=args.split)
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models(checkpoint_name=args.checkpoint_name)

    # Collecting evaluation metrics
    metrics_dict = np.load(os.path.join(args.checkpoint_dir, 'scores_dict.npy'), allow_pickle=True).item()

    # Print numerical outputs and save to a file
    output_dir = os.path.join(args.checkpoint_root, args.output_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(output_file, 'w') as f:
        f.write("Evaluation Metrics:\n")
        for metric, value in metrics_dict.items():
            f.write(f"{metric}: {value}\n")

    # Plotting the metrics
    plt.figure(figsize=(10, 6))
    metrics = ['acc', 'miou', 'mf1', 'iou_0', 'iou_1', 'F1_0', 'F1_1', 'precision_0', 'precision_1', 'recall_0', 'recall_1']
    values = []
    for metric in metrics:
        if metric in metrics_dict:
            values.append(metrics_dict[metric])
        else:
            values.append(0)  # Default value if key is missing
    plt.bar(metrics, values, color=['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'yellow'])
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Evaluation Metrics')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, 'evaluation_metrics_plot.png'))  # Save plot as image
    plt.show()

if __name__ == '__main__':
    main()
