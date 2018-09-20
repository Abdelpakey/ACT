import argparse

# parameters setting
def paras_init():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_frames',
                        type = int,
                        default=8,
                        help = 'batch_frames')
    parser.add_argument('--batchSize',
                        type = int,
                        default = 128,
                        help = 'train.batchSize')
    parser.add_argument('--batch_pos',
                        type = int,
                        default= 32,
                        help = 'batch_pos')
    parser.add_argument('--batch_neg',
                        type= int,
                        default= 96,
                        help='batch_neg')
    parser.add_argument('--train_useGpu',
                        type=bool,
                        default=True,
                        help='train.useGpu')
    parser.add_argument('--train_conserveMemory',
                        type=bool,
                        default=True,
                        help='conserveMemory')
    parser.add_argument('--train_learningRate',
                        type=int,
                        default=0.0001,
                        help='learningRate')
    parser.add_argument('--sampling_posRange',
                        type=list,
                        default=[0.7,1],
                        help='sampling_posRange')
    parser.add_argument('--sampling_negRange',
                        type=list,
                        default=[0,0.5],
                        help='sampling_negRange')
    parser.add_argument('--sampling_input_size',
                        type=int,
                        default=107,
                        help='sampling_input_size')
    parser.add_argument('--sampling_crop_padding',
                        type=int,
                        default=1.1,
                        help='sampling_crop_padding')
    parser.add_argument('--sampling_posPerFrame',
                        type=int,
                        default=50,
                        help='sampling_posPerFrame')
    parser.add_argument('--sampling_negPerFrame',
                        type=int,
                        default=200,
                        help='sampling_negPerFrame')
    parser.add_argument('--sampling_scale_factor',
                        type=int,
                        default=1.05,
                        help='sampling_scale_factor')
    parser.add_argument('--cyclenum',
                        type=int,
                        default=100,
                        help='cyclenumber')

    args = parser.parse_args()
    return args