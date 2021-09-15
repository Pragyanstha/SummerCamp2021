import configargparse

def parse(args = None):
    parser = configargparse.ArgParser()
    parser.add('-c', '--my-config', required=True, is_config_file=True, help='config file path')
    parser.add('--image_size', type=int, default= 32 , help='Size of image for discriminator input.')
    parser.add('--data_dir', type=str, default= 'data/preprocessed' , help='Path to datatset')
    parser.add('--initial_size', type=int, default=8 , help='Initial size for generator.')
    parser.add('--patch_size', type=int, default=4 , help='Patch size for generated image.')
    parser.add('--num_classes', type=int, default=1 , help='Number of classes for discriminator.')
    parser.add('--lr_gen', type=float, default=0.0001 , help='Learning rate for generator.')
    parser.add('--lr_dis', type=float, default=0.0001 , help='Learning rate for discriminator.')
    parser.add('--weight_decay', type=float, default=1e-3 , help='Weight decay.')
    parser.add('--latent_dim', type=int, default=1024 , help='Latent dimension.')
    parser.add('--n_critic', type=int, default=5 , help='n_critic.')
    parser.add('--max_iter', type=int, default=500000 , help='max_iter.')
    parser.add('--gener_batch_size', type=int, default=32 , help='Batch size for generator.')
    parser.add('--train_batch_size', type=int, default=32 , help='Batch size for discriminator.')
    parser.add('--epochs', type=int, default=200 , help='Number of epoch.')
    parser.add('--output_dir', type=str, default='checkpoint' , help='Checkpoint.')
    parser.add('--dim', type=int, default=384 , help='Embedding dimension.')
    parser.add('--img_name', type=str, default="img_name" , help='Name of pictures file.')
    parser.add('--optim', type=str, default="Adam" , help='Choose your optimizer')
    parser.add('--loss', type=str, default="wgangp_eps" , help='Loss function')
    parser.add('--phi', type=int, default="1" , help='phi')
    parser.add('--beta1', type=int, default="0" , help='beta1')
    parser.add('--beta2', type=float, default="0.99" , help='beta2')
    parser.add('--lr_decay', type=str, default=True , help='lr_decay')
    parser.add('--diff_aug', type=str, default="translation,cutout,color", help='Data Augmentation')

    if args == None:
        return parser.parse_args()
    else:
        return parser.parse_args(args)