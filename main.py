import os

from config import get_config
from solver2 import Solver2
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import glob
import datetime

def main(config):
    cudnn.benchmark = True  # Improves runtime if the input size is constant

    config.outputs_dir = os.path.join('experiments', config.outputs_dir)

    config.log_dir = os.path.join(config.outputs_dir, config.log_dir)
    config.model_save_dir = os.path.join(
        config.outputs_dir, config.model_save_dir)
    config.sample_dir = os.path.join(config.outputs_dir, config.sample_dir)
    config.result_dir = os.path.join(config.outputs_dir, config.result_dir)

    # data_loader_2 = get_loader(config.image_dir, config.attr_path, config.c_dim,
    #                               config.image_size, config.AU_batch_size, config.mode,
    #                               config.num_workers, True)

#The vars() function returns the __dic__ attribute of an object. 
#The __dict__ attribute is a dictionary containing the object's changeable attributes.
    config_dict = vars(config)
    # solver2= Solver2(data_loader_2,config_dict)

    if config.mode == 'train':
        initialize_train_directories(config)
        # solver2.train()
        # del solver2
        data_loader = get_loader(config.image_dir, config.attr_path, config.c_dim,
                             config.image_size, config.batch_size, config.mode,
                             config.num_workers, False)
        
        while(True):
            # current_time = datetime.datetime.now().hour
            # if current_time< 22 and current_time>= 10 :
            #     config.run_device='cuda'
            # else:
            #     config.run_device='cpu'
            config_dict=vars(config)
            solver = Solver(data_loader, config_dict)
            try:
                solver.train()
                break
            except RuntimeError:
                model_dir = glob.glob('/home/mtech/kmeher/Full_Model/experiments/experiment1/models/*') 
                latest_file = max(model_dir, key=os.path.getctime)
                iter_no, epoch, _ = latest_file.split('-')
                config.resume_iters= int(iter_no.split('/')[-1])
                config.run_device='cpu'
                config.first_epoch=int(epoch)-300
                config_dict=vars(config)
                del solver

    elif config.mode == 'animation':
        initialize_animation_directories(config)
        data_loader = get_loader(config.image_dir, config.attr_path, config.c_dim,
                             config.image_size, config.batch_size, config.mode,
                             config.num_workers, False)
        solver = Solver(data_loader, config_dict)
        solver.animation()
        
    elif config.mode == 'Fine_Tune':
        data_loader = get_loader(config.image_dir, config.attr_path, config.c_dim,
                             config.image_size, config.batch_size, config.mode,
                             config.num_workers, False)
        solver = Solver(data_loader, config_dict)
        solver.finetune()


def initialize_train_directories(config):
    if not os.path.exists('experiments'):
        os.makedirs('experiments')
    if not os.path.exists(config.outputs_dir):
        os.makedirs(config.outputs_dir)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)


def initialize_animation_directories(config):
    if not os.path.exists(config.animation_results_dir):
        os.makedirs(config.animation_results_dir)


if __name__ == '__main__':

    config = get_config()
    print(config)

    main(config)
