import subprocess
import torch.multiprocessing as mp
import os
import yaml
import argparse
import socket
import time
from time import sleep
from tests_to_run import *
# from prompt_toolkit.contrib.telnet import TelnetServer
# from extra_util.system_info import SystemInfo

"""
Naming convention:
the input to the code:
test name: wild_024_segnet_1, (dataset)_(used_views)_(used_model)_(model_version)_(back_bone)

the output:
folder with the name
example: (train, test)_(dataset)_(used_views)_(used_model)_(model_version)_(back_bone)  
"""


def get_folder_with_smallest_val_loss(root_folder):
    smallest_loss = float('inf')
    smallest_folder = None

    # Iterate through the folders in the root directory
    for folder_name in os.listdir(root_folder):
        try:
            # Split the folder name and find the val_loss part (e.g., '4.35' from 'val_loss=4.35')
            parts = folder_name.split('val_loss=')  # Split at 'val_loss='
            if len(parts) > 1:
                val_loss_str = parts[1].split('.ckpt')[0]  # Extract the number part before '.ckpt'
                val_loss = float(val_loss_str)

                # Compare and track the smallest val_loss
                if val_loss < smallest_loss:
                    smallest_loss = val_loss
                    smallest_folder = folder_name
        except ValueError:
            # If conversion to float fails, ignore this folder
            continue

    if smallest_folder:
        return os.path.join(root_folder, smallest_folder)
    else:
        return None


def change_folder_name(test_name, console_output=None, is_training=True):
    """ change the last changed folder in lightning_logs to test_name """
    if is_training:
        from_directory = 'lightning_logs'
        to_directory = MODELS_FOLDER
    else:
        from_directory = 'lightning_logs'
        to_directory = TESTS_MODELS

    # Get a list of all directories in the given directory
    from_dirs = [d for d in os.listdir(from_directory) if os.path.isdir(os.path.join(from_directory, d))]

    # Sort the directories based on modification time
    from_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(from_directory, x)), reverse=True)

    # the most recently modified
    folder_to_change = os.path.join(from_directory, from_dirs[0]) if from_dirs else None

    # Get a list of all directories in the to_directory
    to_dirs = [d for d in os.listdir(to_directory) if os.path.isdir(os.path.join(to_directory, d))]

    # change folder name
    counter = 2
    new_test_name = test_name
    while new_test_name in to_dirs:
        new_test_name = test_name + "-" + str(counter)
        counter += 1
        if counter >= 50:  # to stop the loop if there is an error
            break
    test_name = new_test_name

    new_name = os.path.join(to_directory, test_name)
    os.rename(folder_to_change, new_name)

    if console_output:
        clean_console_output = [line for line in console_output.splitlines() if not line.startswith('Testing')]
        clean_console_output.insert(0, '=' * len('-----   ' + test_name + '   -----'))
        clean_console_output.insert(1, '-----   ' + test_name + '   -----')
        clean_console_output.insert(2, '=' * len('-----   ' + test_name + '   -----'))

        with open(os.path.join(new_name, "console_output.txt"), "w") as file:
            file.write(str('\n'.join(clean_console_output)))

    return folder_to_change, test_name


def command_training(test_name, train_config, model_config, data_config, auxiliary_config):
    print('\n---------- Training model ', test_name, '\n')
    script_path = 'main.py'
    # Convert to the format used by os.path
    train_config = os.path.normpath(train_config)
    model_config = os.path.normpath(model_config)
    data_config = os.path.normpath(data_config)
    auxiliary_config = os.path.normpath(auxiliary_config)

    if PRINT_COMMAND:
        print('python', script_path, 'fit',
              '-c', os.path.join(f'configs', str(train_config) + '.yml'),
              '-c', os.path.join(f'configs', str(data_config) + '.yml'),
              '-c', os.path.join(f'configs', str(model_config) + '.yml'),
              '-c', os.path.join(f'configs', str(OS_CONFIG) + '.yml'),
              #'-c', os.path.join(f'configs', str(auxiliary_config) + '.yml'),
              )

    subprocess.run(['python', script_path, 'fit',
                    '-c', os.path.join(f'configs', str(train_config) + '.yml'),
                    '-c', os.path.join(f'configs', str(data_config) + '.yml'),
                    '-c', os.path.join(f'configs', str(model_config) + '.yml'),
                    '-c', os.path.join(f'configs', str(OS_CONFIG) + '.yml'),
                    #'-c', os.path.join(f'configs', str(auxiliary_config) + '.yml'),
                    # '-c', auxiliary_commands,
                    ])
    # need sleep after each subprocess
    sleep(2)

    folder_to_change, test_name = change_folder_name('train_' + test_name)
    return test_name


def command_testing(test_name, model_folder_path=None):
    print('\n---------- Testing model ', test_name, '\n')
    script_path = 'main.py'
    models_folder = MODELS_FOLDER
    if model_folder_path is not None:
        models_folder = MODELS_FOLDER + '/' + model_folder_path

    # enshoure the model is saved and exsts
    config_path = f'{models_folder}/train_{test_name}'
    with open(config_path + '/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    model_path = get_folder_with_smallest_val_loss(f'{models_folder}/train_{test_name}/checkpoints')

    print('best model ', model_path)

    if PRINT_COMMAND:
        command = 'python ' + script_path + ' test ' + ' -c ' + config_path + '/config.yaml' + ' --ckpt ' + f'{model_path}'
        # print(command)
        command = command.replace("Data 1", r"Data\ 1")
        # print(command)
        print('python', script_path, 'test', '-c', f'{MODELS_FOLDER}/train_{test_name}/config.yaml',
              '--ckpt', model_path)

    results = subprocess.run(['python', script_path, 'test',
                              '-c', f'{models_folder}/train_{test_name}/config.yaml',
                              '--ckpt', model_path,
                              ],
                             capture_output=True, text=True
                             )

    console_output = None
    if results.returncode == 0:
        console_output = results.stdout
    # need sleep after each subprocess
    sleep(2)
    try:
        change_folder_name('Assessing_' + test_name, console_output=console_output, is_training=False)
    except:
        change_folder_name('Assessing_Holder_' + test_name, console_output=console_output, is_training=False)


def testing_on_different_views(test_name, views, model_folder_path=None):
    print('\n---------- Testing model on different views', test_name, 'views:', views, '\n')

    models_folder = MODELS_FOLDER
    if model_folder_path is not None:
        models_folder = MODELS_FOLDER + '/' + model_folder_path

    # crate new yml with the new test_views
    config_path = f'{models_folder}/train_{test_name}'

    # Load the YAML file
    with open(config_path + '/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Modify the data (example: changing a value)
    config['data']['init_args']['test_cameras'] = views

    # Save the modified data back to the YAML file
    with open(config_path + '/config_2.yaml', 'w') as file:
        yaml.safe_dump(config, file)

    script_path = 'main.py'

    model_path = get_folder_with_smallest_val_loss(f'{config_path}/checkpoints')
    if PRINT_COMMAND:
        command = 'python ' + script_path + ' test ' + '-c ' + config_path + '/config_2.yaml ' + '--ckpt ' + f'{model_path}'
        command.replace("Data 1", "Data\ 1")
        print(command)
        # print('python', script_path, 'test', '-c', config_path + '/config_2.yaml',
        #       '--ckpt', f'{config_path}/checkpoints/last.ckpt')

    # model_path = get_folder_with_smallest_val_loss(f'{config_path}/checkpoints')
    results = subprocess.run(['python', script_path, 'test',
                              '-c', config_path + '/config_2.yaml',
                              '--ckpt', model_path],
                             capture_output=True, text=True
                             )
    console_output = None
    if results.returncode == 0:
        console_output = results.stdout

    # need sleep after each subprocess
    sleep(2)
    views_together = ''.join(map(str, views))

    try:
        change_folder_name('Assessing_' + test_name + f'_{views_together}', console_output=console_output,
                           is_training=False)
    except:
        change_folder_name('Assessing_Holder_' + test_name + f'_{views_together}', console_output=console_output,
                           is_training=False)


def config_files_exists(tests_dic):
    for test_name in tests_dic.keys():
        conf_files = tests_dic[test_name]
        for conf_file in conf_files:
            if type(conf_file) is str:
                conf_path = f'configs/{conf_file}.yml'
                if os.path.exists(conf_path):
                    pass
                else:
                    print(conf_file, 'dose not exist')
                    exit()


if __name__ == '__main__':
    pc_name = os.getenv('COMPUTERNAME') or os.getenv('HOSTNAME') or socket.gethostname()
    if pc_name == 'ipi8':
        """
        MODELS_FOLDER: where i save the trained models
        TESTS_MODELS: where i save the testes of each model, i usualy test the models on 2 diffrent camera views
        """
        MODELS_FOLDER = '/home/s-jiang/Documents/Robust-MVMOT/World-track/model_weights/'
        TESTS_MODELS = '/home/s-jiang/Documents/Robust-MVMOT/World-track/model_test/'
        OS_CONFIG = "os_linux"
        tests = tests_129_average_res18 # tests_8 tests_129_average_res50

    experiment_name = 'res18_fpn_v2'  # 'use_pretrained_MaskedRcnn'# 'averagePool_2DMask_3'  # 'averagePool_2DMask_2'# 'concate_avrage' local_retrained_experiment coco_pretrained_experiment
    def list_of_strings(arg):
        return arg.split(',')

    parser = argparse.ArgumentParser()
    parser.add_argument("-mn", "--model_name", help="a dict of the models to train", default=None)
    parser.add_argument("-en", "--experiment_name",
                        help="name of experiment and the folder where it will be saved",
                        default=experiment_name)
    parser.add_argument("-nr", "--experiment_repetition", type=int,
                        help="number of experiments repetition",
                        default=1)
    parser.add_argument("--add_suffix",
                        help="add somthing to the begining of test name",
                        default=None)
    parser.add_argument('-mnl', '--model_names_list', type=list_of_strings,
                        help="pass multiple model_name dict and there repetition, "
                             "the names are seperated by ',' ... test_1,2,test_2,3,test_3,1",
                        default=None)
    parser.add_argument("-t", "--tests", 
                    help="specify which test configuration to use",)
    parser.add_argument("--additional_args", default=None)
    args = parser.parse_args()

    if args.tests is not None:
        print(f"📋 Using test configuration from command line: {args.tests}")
        # 根据传入的字符串选择对应的测试配置
        if args.tests == 'tests_8_1':
            tests = tests_8_1
        elif args.tests == 'tests_8_2':
            tests = tests_8_2
        elif args.tests == 'tests_8':
            tests = tests_8
        elif args.tests == 'tests_129_average_1':
            tests = tests_129_average_1
        elif args.tests == 'tests_129_average_2':
            tests = tests_129_average_2
        elif args.tests == 'tests_129_average_3':
            tests = tests_129_average_3
        elif args.tests == 'tests_129_average_res18':
            tests = tests_129_average_res18
        elif args.tests == 'tests_129_average_res50':
            tests = tests_129_average_res50
        else:
            print(f"⚠️ Warning: Unknown test configuration '{args.tests}', using default")
            # 保持使用默认的tests配置
    else:
        print(f"📋 Using default test configuration: tests_129_average_res18_1")

    # 处理model_name参数（保持原有逻辑，但现在优先级低于-t参数）
    if args.model_name is not None and args.tests is None:
        print(f"📋 Using test configuration from model_name: {args.model_name}")
        if args.model_name == 'tests_8_1':
            tests = tests_8_1
        if args.model_name == 'tests_8_2':
            tests = tests_8_2

    # repeat the rest n times
    tests = add_suffix_to_test_name(tests, args.add_suffix)

    # repeat the rest n times
    tests = repeat_test_name(tests, args.experiment_repetition)

    if args.model_names_list is not None:
        # repeat the rest n times
        tests = mearg_multiple_test(args.model_names_list)

    for key in tests.keys():
        print(key)

    experiment_name = args.experiment_name
    print(f"Running on PC: {pc_name}")
    print(f"🎯 Selected test configuration: {type(tests).__name__ if hasattr(tests, '__name__') else 'custom_dict'}")
    
    # print all the commands
    PRINT_COMMAND = True
    MODELS_FOLDER = MODELS_FOLDER + experiment_name
    TESTS_MODELS = TESTS_MODELS + experiment_name

    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

    if not os.path.exists(TESTS_MODELS):
        os.makedirs(TESTS_MODELS)

    # enssure that all the file pathes are corect
    config_files_exists(tests)

    tests_length = len(tests)
    problem_runs = []
    # run the trainings and tests
    for i, test_name in enumerate(tests.keys()):
        start_time = time.time()

        print('\n\n ----------------------------------------------------------------')
        print(f'Experiments: {experiment_name}\nMODEL {i + 1} of {tests_length}')

        try:
            conf_files = tests[test_name]
            print('working on: ', test_name)
            # command_training(test_name, conf_files[0], conf_files[1], conf_files[2], conf_files[4])
            command_testing(test_name)
            testing_on_different_views(test_name, conf_files[3])

        except:
            problem_runs.append(test_name)
            print(f'problems with {test_name} \n and \n')
            for i in problem_runs: print(i)

        fin_time = time.time()
        print('finished: ', time.strftime('%H:%M:%S'))
        running_time = round((fin_time - start_time) / 60, 2)
        print('running for: ', running_time, ' minutes')
