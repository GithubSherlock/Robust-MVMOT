import os.path

import matplotlib.pyplot as plt
import pandas as pd

# 2 views
test_to_compare_1 = ['wild_02_mvdet_2_res18', 'wild_02_segnet_1_res18_Z8', 'wild_02_segnet_2_res18_Z8',
                     #'wild_02_segnet_3_res18_Z8', 'wild_02_segnet_4_res18_Z8'
                     ]

test_to_compare_2 = ['wild_13_mvdet_2_res18', 'wild_13_segnet_1_res18_Z8', 'wild_13_segnet_2_res18_Z8',
                     #'wild_13_segnet_3_res18_Z8', 'wild_13_segnet_4_res18_Z8'
                     ]

test_to_compare_3 = ['wild_25_mvdet_2_res18', 'wild_25_segnet_1_res18_Z8', 'wild_25_segnet_2_res18_Z8',
                     #'wild_25_segnet_3_res18_Z8', 'wild_25_segnet_4_res18_Z8'
                     ]

# 3 views
test_to_compare_4 = ['wild_026_mvdet_1_res18',  # 'wild_026_mvdet_2_res18',
                     'wild_026_segnet_1_res18_Z8',
                     # 'wild_026_segnet_1_res18',
                     'wild_026_segnet_2_res18',
                     'wild_026_segnet_4_res18_Z8']

test_to_compare_5 = ['wild_135_mvdet_1_res18',  # 'wild_135_mvdet_2_res18',
                     'wild_135_segnet_1_res18_Z8',
                     # 'wild_135_segnet_1_res18',
                     'wild_135_segnet_2_res18',
                     #'wild_135_segnet_4_res18_Z8'
                     ]

# 4 views
test_to_compare_6 = ['wild_0246_mvdet_1_res18', 'wild_0246_segnet_1_res18', 'wild_0246_segnet_1_res18_Z8',
                     'wild_0246_segnet_2_res18', 'wild_0246_segnet_2_res18_Z8']

test_to_compare_7 = ['wild_1345_mvdet_1_res18', 'wild_1345_mvdet_2_res18', 'wild_1345_segnet_1_res18',
                     'wild_1345_segnet_2_res18']

# number of Z
test_to_compare_8 = ['wild_135_segnet_1_res18', 'wild_135_segnet_1_res18_Z4', 'wild_135_segnet_1_res18_Z8',
                     'wild_135_segnet_1_res18_Z16']

FILE_PATH = 'util_outputs/simplified_df.xlsx'


def get_test_on_diff_views(df, test_name):
    tests_list = df['Test Name']
    filtered_names = list(filter(lambda name: name.startswith(test_name), tests_list))
    # print(test_name, filtered_names)
    return filtered_names


def get_train_test_views(test_name, is_test=False):
    tests_name_split = test_name.split('_')
    train_views = tests_name_split[1]
    test_views = train_views
    if is_test:
        test_views = tests_name_split[-1]

    train_views = '[' + ','.join([char for char in train_views]) + ']'
    test_views = '[' + ','.join([char for char in test_views]) + ']'
    return train_views, test_views


def get_scatter(df, test_to_compare, metric='detect/precision'):
    save_path = '/media/rasho/Data 1/Arbeit/saved_models/temp_saver/earlyBird_results/numeric'

    group = [get_test_on_diff_views(df, test) for test in test_to_compare]
    group = [sublist for sublist in group if sublist]

    groups_ = list(map(list, zip(*group)))
    scater_1 = df[df['Test Name'].isin(groups_[0])][metric]
    scater_2 = df[df['Test Name'].isin(groups_[1])][metric]

    # for temp in groups_[0]:
    #     print(temp, df[df['Test Name'].isin([temp])][metric])

    # models = ['mvdet_1', 'mvdet_2', 'segnet_1', 'segnet_2', 'segnet_3', 'segnet_4']
    if len(groups_[0]) == 5:
        X = ['mvdet', 'segnet_1', 'segnet_2', 'segnet_3', 'segnet_4']
    elif len(groups_[0]) == 15:
        X_jiter_o = [i for i in range(5)] * 3
        X_jiter = [0, 1, 2, 3, 4,
                   0.2, 1.2, 2.2, 3.2, 4.2,
                   0.4, 1.4, 2.4, 3.4, 4.4]
        X = ['mvdet', 'segnet_1', 'segnet_2', 'segnet_3', 'segnet_4',
             'mvdet', 'segnet_1', 'segnet_2', 'segnet_3', 'segnet_4',
             'mvdet', 'segnet_1', 'segnet_2', 'segnet_3', 'segnet_4']

    elif len(groups_[0]) == 8:
        X_jiter_o = [i for i in range(4)] * 2
        X_jiter = [0, 1, 2, 3,
                   0.2, 1.2, 2.2, 3.2,
                   ]
        X = ['mvdet', 'segnet_1', 'segnet_2', 'segnet_4',
             'mvdet', 'segnet_1', 'segnet_2', 'segnet_4', ]

    elif len(groups_[0]) == 4:
        X_jiter_o = [i for i in range(4)] * 1
        X_jiter = [0, 1, 2, 3,
                   # 0.2, 1.2, 2.2, 3.2,
                   ]
        X = ['Z=1', 'Z=4', 'Z=8', 'Z=18',
             # 'Z=1', 'Z=4', 'Z=8', 'Z=18'
             ]
    # X = groups_[0]  # ['mvdet', 'segnet_1', 'segnet_2', 'segnet_3', 'segnet_4']
    print(len(X_jiter), len(scater_1))
    # Create scatter plot
    train_views_1, test_views_1 = get_train_test_views(groups_[1][0], is_test=True)
    # train_views_2, test_views_2 = get_train_test_views(groups_[1][6], is_test=True)
    # train_views_3, test_views_3 = get_train_test_views(groups_[1][4], is_test=True)

    plt.figure(figsize=(8, 6))  # Optional: Adjust figure size
    start = 0
    end = 4
    plt.scatter(X_jiter[start:end], scater_1[start:end], color='blue', marker='o',
                label=f'{train_views_1}/{train_views_1}', s=50)
    plt.scatter(X_jiter[start:end], scater_2[start:end], color='red', marker='o',
                label=f'{train_views_1}/{test_views_1}', s=50)

    # start = 5
    # end = 10
    # plt.scatter(X_jiter[start:end], scater_1[start:end], color='blue', marker='.',
    #             label=f'{train_views_2}/{train_views_2}', s=50)
    # plt.scatter(X_jiter[start:end], scater_2[start:end], color='red', marker='.',
    #             label=f'{train_views_2}/{test_views_2}', s=50)

    # start = 4
    #
    # plt.scatter(X_jiter[start:], scater_1[start:], color='blue', marker='*',
    #             label=f'{train_views_3}/{train_views_3}', s=50)
    # plt.scatter(X_jiter[start:], scater_2[start:], color='red', marker='*',
    #             label=f'{train_views_3}/{test_views_3}', s=50)

    plt.xlim([-1, 5])
    # plt.ylim([0, 500])

    plt.xticks(X_jiter_o, X)
    # Customize plot
    # plt.title('Scatter Plot of Test Data')
    plt.xlabel('Used Model', fontweight='bold')
    plt.ylabel(metric.split('/')[1], fontweight='bold')
    plt.legend(title='Training views/ \nTesting Views')
    plt.grid(True)

    # # Show plot
    plt.show()
    # plot_name = '_'.join(test_to_compare[0].split('-')[:2]) + '_' + metric.split('/')[1]
    #
    # save_name = f'{plot_name}.png'
    # plt.savefig(os.path.join(save_path, save_name), dpi=300)  # Replace 'my_plot.png' with your desired filename
    #
    # # (Optional) Close the plot window if you don't need it displayed
    # plt.close()


def get_line(df, test_to_compare, metric='detect/precision'):
    save_path = '/media/rasho/Data 1/Arbeit/saved_models/temp_saver/earlyBird_results/numeric'

    group = [get_test_on_diff_views(df, test) for test in test_to_compare]
    group = [sublist for sublist in group if sublist]

    # scater_1 = df[df['Test Name'].isin(groups_[0])][metric]
    # scater_2 = df[df['Test Name'].isin(groups_[1])][metric]

    train_test_to_show = ['mvdet', 'segnet_1', 'segnet_2',# 'segnet_3', 'segnet_4',
                          'mvdet', 'segnet_1', 'segnet_2',# 'segnet_3', 'segnet_4',
                          'mvdet', 'segnet_1', 'segnet_2',# 'segnet_3', 'segnet_4',
                          'mvdet', 'segnet_1', 'segnet_2',# 'segnet_4',
                          #'mvdet', 'segnet_1', 'segnet_2',# 'segnet_4',
                          ]

    x_offsets = dict(zip(['mvdet', 'segnet_1', 'segnet_2'],#, 'segnet_3', 'segnet_4'],
                         [0, 0.12, 0.24,]# 0.36, 0.48]
                         ))
    colors = dict(zip(['mvdet', 'segnet_1', 'segnet_2'],#, 'segnet_3', 'segnet_4'],
                      ['r', 'b', 'g'],# , 'c', 'k']
                      ))
    Xs = [item for r in [[0] * 3, [1] * 3, [2] * 3, [3] * 3] for item in r]

    train_test_views = []
    for index, tests in enumerate(group):
        model_name = train_test_to_show[index]
        x_offset = x_offsets[model_name]
        color = colors[model_name]

        X = [Xs[index] + x_offset, Xs[index] + x_offset]
        metric_score = df[df['Test Name'].isin(tests)][metric][:2]

        if model_name == 'mvdet':
            train_views, test_views = get_train_test_views(tests[1], is_test=True)
            train_test_views.append(f'{train_views}\n{test_views}')

        line_style = '-'  # Choose any line style from Matplotlib documentation (e.g., '--', ':')
        marker_style = 'o'  # Choose any marker style (e.g., 's' for square, '^' for triangle)
        markersize = 10
        s = 300
        if Xs[index] == 0:  # add lable just once per test
            plt.plot(X, metric_score, linestyle=line_style,
                     marker=marker_style, markersize=markersize, color=color,
                     label=f'{model_name}'
                     )  # Adjust marker size (optional)

            # plt.scatter(X[1], metric_score.iloc[1],
            #          marker='*', s=s, color=color)

        else:
            plt.plot(X, metric_score, linestyle=line_style,
                     marker=marker_style, markersize=markersize, color=color,
                     )  # Adjust marker size (optional)

            # plt.scatter(X[1], metric_score.iloc[1],
            #          marker='*', s=s, color=color)

        x_offset += 0.2

    plt.xticks(range(4), train_test_views)
    # Customize plot
    # plt.title('Scatter Plot of Test Data')
    plt.xlim([-2, 5])
    plt.xlabel('Used Train/Test views', fontweight='bold')
    plt.ylabel(metric.split('/')[1] + '[%]', fontweight='bold')
    plt.legend(title='Model')
    plt.grid(True)

    # plt.show()

    plot_name = '_'.join(test_to_compare[0].split('-')[:2]) + '_' + metric.split('/')[1]
    save_name = f'{plot_name}.png'
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, save_name), dpi=300)  # Replace 'my_plot.png' with your desired filename
    plt.close()



if __name__ == "__main__":
    df = pd.read_excel(FILE_PATH)
    df = df.dropna()

    test_to_compare_l_1 = test_to_compare_1 + test_to_compare_2 + test_to_compare_3
    test_to_compare_l_2 = test_to_compare_5  # + test_to_compare_3
    metrics = ['detect/precision', 'detect/recall', 'track/idf1', 'track/mota', 'detect/moda', 'detect/modp']
    for i in range(len(metrics)):
        get_line(df, test_to_compare_l_1 + test_to_compare_l_2, metric=metrics[i])
        # get_scatter(df, test_to_compare_l_1, metric=metrics[i])

    exit()
    tests_to_compare = [test_to_compare_1, test_to_compare_2, test_to_compare_3,
                        test_to_compare_4, test_to_compare_5]
    # print(get_test_on_diff_views(df, test))
    # exit()
    for test_to_compare in tests_to_compare:
        metrics = ['detect/precision', 'detect/recall', 'track/idf1', 'track/mota', 'track/motp']
        for i in range(len(metrics)):
            get_scatter(df, test_to_compare, metric=metrics[i])
