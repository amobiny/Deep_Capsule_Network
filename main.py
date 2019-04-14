import tensorflow as tf
from config import args
import os
from utils import write_spec

if args.model == 'original_capsule':
    from models.Original_CapsNet import Orig_CapsNet as Model
elif args.model == 'matrix_capsule':
    from models.Matrix_Capsule_EM_routing import MatrixCapsNet as Model
elif args.model == 'fast_capsule':
    from models.Fast_CapsNet import FastCapsNet as Model
elif args.model == 'vector_capsule':
    from models.Deep_CapsNet import CapsNet as Model
elif args.model == 'alexnet':
    from models.AlexNet import AlexNet as Model
elif args.model == 'resnet':
    from models.ResNet import ResNet as Model
elif args.model == 'densenet':
    from models.DenseNet import DenseNet as Model


def main(_):
    if args.mode not in ['train', 'train_sequence', 'test', 'test_sequence',
                         'get_features', 'grad_cam', 'grad_cam_sequence']:
        print('invalid mode: ', args.mode)
        print("Please input a mode: train or test")
    elif args.mode == 'train' or args.mode == 'test' or args.mode == 'get_features' or args.mode == 'grad_cam':
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

    elif args.mode == 'train_sequence' or args.mode == 'test_sequence' or args.mode == 'grad_cam_sequence':
        session = tf.Session()
        CNN_Model = Model(session, args)
        from models.Recurrent_Network import RecNet
        model = RecNet(session, args, CNN_Model)
        if not os.path.exists(args.rnn_modeldir+args.rnn_run_name):
            os.makedirs(args.rnn_modeldir+args.rnn_run_name)
        if not os.path.exists(args.rnn_logdir+args.rnn_run_name):
            os.makedirs(args.rnn_logdir+args.rnn_run_name)
        if args.mode == 'train_sequence':
            write_spec(args)
            model.train()
        elif args.mode == 'test_sequence':
            model.test(args.reload_step, args.rnn_reload_step)
        elif args.mode == 'grad_cam_sequence':
            if not os.path.exists(args.imgdir + args.rnn_run_name):
                os.makedirs(args.imgdir + args.rnn_run_name)
            model.grad_cam(args.reload_step, args.rnn_reload_step)


if __name__ == '__main__':
    # configure which gpu or cpu to use
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    tf.app.run()
