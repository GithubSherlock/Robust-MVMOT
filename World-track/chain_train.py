import subprocess
from time import sleep
from pathlib import Path
import torch.multiprocessing as mp
import os
import yaml
from icecream import ic
from prompt_toolkit.contrib.telnet import TelnetServer
from extra_util.system_info import SystemInfo

"""
Naming convention:
the input to the code:
test name: wild_024_segnet_1, (dataset)_(used_views)_(used_model)_(model_version)_(back_bone)

the output:
folder with the name
example: (train, test)_(dataset)_(used_views)_(used_model)_(model_version)_(back_bone)  
"""

if SystemInfo.get_os_info()['Operating system'] == 'Linux':
    OS_CONFIG = "os_linux"
    """
    MODELS_FOLDER: where i save the trained models
    TESTS_MODELS: where i save the testes of each model, i usualy test the models on 2 diffrent camera views
    """
    MODELS_FOLDER = '/home/deep/PythonCode/EarlyBird/World_track-main/model_weights'
    TESTS_MODELS = '/home/deep/PythonCode/EarlyBird/World_track-main/tests'
elif SystemInfo.get_os_info()['Operating system'] == 'Windows':
    OS_CONFIG = "os_windows"
    MODELS_FOLDER = ''
    TESTS_MODELS = ''
elif SystemInfo.get_os_info()['Operating system'] == 'Darwin':
    MODELS_FOLDER = ''
    TESTS_MODELS = ''
    OS_CONFIG = "os_macos"

# OS_Windows = False
# if os.name == 'nt':  # the code is running on windows
#     OS_Windows = True

# if OS_Windows:
#     MODELS_FOLDER = 'D:\Arbeit\models\EarlyBird_models'
#     TESTS_MODELS = 'D:\Arbeit\models\EarlyBird_tests'
#     OS_CONFIG = "Windows"
# else:
#     MODELS_FOLDER = '/media/rasho/M2_Samsung990/Work/Models/EarlyBird/models'
#     TESTS_MODELS = '/media/rasho/M2_Samsung990/Work/Models/EarlyBird/tests'
#     OS_CONFIG = "Linux"

# print(OS_Windows)


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
        new_test_name = test_name + " (" + str(counter) + ")"
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

    return folder_to_change


def command_training(test_name, train_config, model_config, data_config):
    print('\n---------- Training model ', test_name, '\n')
    script_path = 'main.py'
    # Convert to the format used by os.path
    train_config = os.path.normpath(train_config)
    model_config = os.path.normpath(model_config)
    data_config = os.path.normpath(data_config)

    if PRINT_COMMAND:
        print('python', script_path, 'fit',
              '-c', os.path.join(f'configs', str(train_config) + '.yml'),
              '-c', os.path.join(f'configs', str(data_config) + '.yml'),
              '-c', os.path.join(f'configs', str(model_config) + '.yml'),
              '-c', os.path.join(f'configs', str(OS_CONFIG) + '.yml'))

    subprocess.run(['python', script_path, 'fit',
                    '-c', os.path.join(f'configs', str(train_config) + '.yml'),
                    '-c', os.path.join(f'configs', str(data_config) + '.yml'),
                    '-c', os.path.join(f'configs', str(model_config) + '.yml'),
                    '-c', os.path.join(f'configs', str(OS_CONFIG) + '.yml'),
                    ])
    # need sleep after each subprocess
    sleep(2)

    change_folder_name('train_' + test_name)


def command_testing(test_name):
    print('\n---------- Testing model ', test_name, '\n')
    script_path = 'main.py'

    # enshoure the model is saved and exsts
    config_path = f'{MODELS_FOLDER}/train_{test_name}'
    with open(config_path + '/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    if PRINT_COMMAND:
        command = 'python ' + script_path + ' test ' + ' -c ' + config_path + '/config.yaml' + ' --ckpt ' + f'{config_path}/checkpoints/last.ckpt'
        # print(command)
        command = command.replace("Data 1", r"Data\ 1")
        print(command)
        # print('python', script_path, 'test', '-c', f'{MODELS_FOLDER}/train_{test_name}/config.yaml',
        #       '--ckpt', f'{MODELS_FOLDER}/train_{test_name}/checkpoints/last.ckpt', )

    results = subprocess.run(['python', script_path, 'test',
                              '-c', f'{MODELS_FOLDER}/train_{test_name}/config.yaml',
                              '--ckpt', f'{MODELS_FOLDER}/train_{test_name}/checkpoints/last.ckpt',
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


def testing_on_different_views(test_name, views):
    print('\n---------- Testing model on different views', test_name, 'views:', views, '\n')
    # crate new yml with the new test_views
    config_path = f'{MODELS_FOLDER}/train_{test_name}'

    # Load the YAML file
    with open(config_path + '/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Modify the data (example: changing a value)
    config['data']['init_args']['test_cameras'] = views

    # Save the modified data back to the YAML file
    with open(config_path + '/config_2.yaml', 'w') as file:
        yaml.safe_dump(config, file)

    script_path = 'main.py'

    if PRINT_COMMAND:
        command = 'python ' + script_path + ' test ' + '-c ' + config_path + '/config_2.yaml ' + '--ckpt ' + f'{config_path}/checkpoints/last.ckpt'
        command.replace("Data 1", "Data\ 1")
        print(command)
        # print('python', script_path, 'test', '-c', config_path + '/config_2.yaml',
        #       '--ckpt', f'{config_path}/checkpoints/last.ckpt')

    results = subprocess.run(['python', script_path, 'test',
                              '-c', config_path + '/config_2.yaml',
                              '--ckpt', f'{config_path}/checkpoints/last.ckpt'],
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
    # test_name : [training_config, used_model, dataset_config, testing_views]
    tests = {
        'wild_0246_segnet_maxPool_res18_Z4': ['t_fit', 'model/m_segnet_maxPool',
                                              'wild_configs/d_wildtrack_0246_Z4', [1, 3, 4, 5]],
        'wild_1345_segnet_maxPool_res18_Z4': ['t_fit', 'model/m_segnet_maxPool',
                                              'wild_configs/d_wildtrack_1345_Z4', [0, 2, 4, 6]],

    }
    # print all the commands
    PRINT_COMMAND = True

    # enssure that all the file pathes are corect
    config_files_exists(tests)

    tests_length = len(tests)
    # run the trainings and tests
    for i, test_name in enumerate(tests.keys()):
        # command_testing(test_name)
        # testing_on_different_views(test_name, [0, 1])
        # exit()
        print(f'testing {i + 1} of {tests_length}')

        try:
            conf_files = tests[test_name]
            print('working on: ', test_name)
            command_training(test_name, conf_files[0], conf_files[1], conf_files[2])
            # change_folder_name('train_' + test_name)

            command_testing(test_name)
            testing_on_different_views(test_name, conf_files[3])
        except:
            print(f'problems with {test_name}')
