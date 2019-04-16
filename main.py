import tensorflow as tf
from config import args
import os
from utils import write_spec

if args.model == 'original_capsule':
    from models.Original_CapsNet import Orig_CapsNet as Model
elif args.model == 'deep_capsule':
    from models.Deep_CapsNet import CapsNet as Model
elif args.model == 'alexnet':
    from models.AlexNet import AlexNet as Model
elif args.model == 'resnet':
    from models.ResNet import ResNet as Model
elif args.model == 'densenet':
    from models.DenseNet import DenseNet as Model


def main(_):
    if args.mode not in ['train', 'test', 'grad_cam']:
        print('invalid mode: ', args.mode)
        print("Please input a mode: train or test")
    else:
        model = Model(tf.Session(), args)
        if not os.path.exists(args.modeldir+args.run_name):
            os.makedirs(args.modeldir+args.run_name)
        if not os.path.exists(args.logdir+args.run_name):
            os.makedirs(args.logdir+args.run_name)
        if args.mode == 'train':
            write_spec(args)
            model.train()
        elif args.mode == 'test':
            model.test(args.reload_step)
        elif args.mode == 'get_features':
            model.get_features(args.reload_step)
        elif args.mode == 'grad_cam':
            if not os.path.exists(args.imgdir + args.run_name):
                os.makedirs(args.imgdir + args.run_name)
            model.grad_cam(args.reload_step)


if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    tf.app.run()
