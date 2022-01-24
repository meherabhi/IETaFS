import argparse


def get_config():
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=17,
                        help='dimension of domain labels')
    parser.add_argument('--lmk_dim', type=int, default=68,
                    help='dimension of domain labels')

    parser.add_argument('--image_size', type=int,
                        default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64,
                        help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64,
                        help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6,
                        help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6,
                        help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=160,
                        help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10,
                        help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10,
                        help='weight for gradient penalty')
    parser.add_argument('--lambda_sat', type=float, default=0.1,
                        help='weight for attention saturation loss')
    parser.add_argument('--lambda_smooth', type=float, default=1e-4,
                        help='weight for the attention smoothing loss')

    # Training configuration.
    parser.add_argument('--batch_size', type=int,
                        default=16, help='mini-batch size')
    parser.add_argument('--AU_batch_size', type=int,
                        default=300, help='mini-batch size for AU Training')
    parser.add_argument('--num_epochs', type=int, default=60,
                        help='number of total epochs for training D')
    parser.add_argument('--num_epochs_AU_train', type=int, default=300,
                        help='number of total epochs for training AU')
    parser.add_argument('--num_epochs_decay', type=int, default=320,
                        help='number of epochs for start decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001,
                        help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001,
                        help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=2,
                        help='number of D updates per each G update')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='beta2 for Adam optimizer')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int,
                        default= 2001, help='resume training from this step')
    parser.add_argument('--first_epoch', type=int,
                        default= 25,help='First epoch')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('--run_device', type= str, default='cpu', help='Deafult device to run on')
    parser.add_argument('--use_virtual', type=str2bool, default=False,
                        help='Boolean to decide if we should use the virtual cycle concistency loss')
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'animation', 'Fine_Tune'])
    parser.add_argument('--AU_Train', type=str2bool, default=True,
                        help='Initially whether or not to start with AU_Train')
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--num_sample_targets', type=int, default=4,
                        help="number of targets to use in the samples visualization")

    # Directories.
    parser.add_argument('--predictor_dir', type=str,
                    default='Fine_Tune_data/shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--feature_dir', type=str,
                    default='Fine_Tune_data/features')
    parser.add_argument('--image_dir', type=str,
                        default='data/RAVDESS')
    parser.add_argument('--attr_path', type=str,
                        default='data/RAVDESS/attributes.txt')
    parser.add_argument('--outputs_dir', type=str, default='experiment1')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--model_save_dir', type=str, default='models')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--result_dir', type=str, default='results')

    parser.add_argument('--animation_images_dir', type=str,
                        default='animations/eric_andre/images_to_animate')
    parser.add_argument('--fine_tune_images_dir', type=str,
                        default='Fine_Tune_data/Frames/Real_Frames')
    parser.add_argument('--animation_spectrograms_dir', type=str,
                    default='animations/eric_andre/spectrograms')
    parser.add_argument('--fine_tune_spectrograms_dir', type=str,
                    default='Fine_Tune_data/Spectrograms')
    parser.add_argument('--animation_attribute_images_dir', type=str,
                        default='animations/eric_andre/attribute_images')
    parser.add_argument('--animation_attributes_path', type=str,
                        default='animations/eric_andre/attributes.txt')
    parser.add_argument('--fine_tune_attributes_path', type=str,
                    default='Fine_Tune_data/fine_tune_attributes.txt')
    parser.add_argument('--animation_models_dir', type=str,
                        default='animations/eric_andre/pretrained_models')
    parser.add_argument('--animation_results_dir', type=str,
                        default='animations/eric_andre/results')
    parser.add_argument('--animation_attention_dir', type=str,
                    default='animations/eric_andre/attention_maps')
    parser.add_argument('--animation_regression_dir', type=str,
                default='animations/eric_andre/regression_maps')
    parser.add_argument('--fine_tune_results_dir', type=str,
                        default='Fine_Tune_data/Frames/Fake_Frames')
    parser.add_argument('--animation_mode', type=str, default='animate_image',
                        choices=['animate_image', 'animate_random_batch'])

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=200)
    parser.add_argument('--model_save_step', type=int, default=1000)

    config = parser.parse_args()
    return config

#Returns True if the lowercase form of v is present in the string "true" in any format.
def str2bool(v):
    return v.lower() in ('true')
