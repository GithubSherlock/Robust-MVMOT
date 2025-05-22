# test_name : [training_config, used_model, dataset_config, testing_views]
def repeat_test_name(test_name, repetition=1):
    # Extend the dictionary dynamically
    extended_dict = {}
    for num in range(1, repetition + 1):
        for key, value in test_name.items():
            extended_dict[f"{num}_{key}"] = value
    return extended_dict

# def repeat_test_name(test_name, repetition=1):
#     """直接复制原始测试名称，不添加前缀"""
#     extended_dict = {}
#     for _ in range(repetition):
#         extended_dict.update(test_name.copy())  # 使用 copy() 避免引用问题
#     return extended_dict


def add_suffix_to_test_name(test_name, suffix=None):
    # Extend the dictionary dynamically
    extended_dict = {}
    if suffix is not None:
        for key, value in test_name.items():
            extended_dict[f"{suffix}_{key}"] = value
    else:
        extended_dict = test_name
    return extended_dict


def mearg_multiple_test(tests_list):
    small_lists = []
    for i in range(len(tests_list)):
        item = tests_list[i]
        if not item.isdigit():
            # Check the next item for a number
            if i + 1 < len(tests_list) and tests_list[i + 1].isdigit():
                small_lists.append([item, int(tests_list[i + 1])])
                i += 2  # Skip the next item as it is a number
            else:
                small_lists.append([item, 1])  # Default to 1 if no number follows
                i += 1
        else:
            i += 1  # Skip standalone numbers

    all_tests = {}
    for test in small_lists:
        test_name, repetition = test
        repeated_test = repeat_test_name(globals()[test_name], repetition)
        all_tests.update(repeated_test)

    return all_tests


tests_129 = {

    'fit1_wild_1345_segnet_averagePool_res18_Z4': ['t_fit',
                                                   'model/m_segnet_averagePool',
                                                   'wild_configs/d_wildtrack_1345_Z4', [0, 2, 4, 6],
                                                   'auxiliary/no_auxiliary'],
    'fit1_wild_0246_segnet_averagePool_res18_Z4': ['t_fit',
                                                   'model/m_segnet_averagePool',
                                                   'wild_configs/d_wildtrack_0246_Z4', [1, 3, 4, 5],
                                                   'auxiliary/no_auxiliary'],

    'fit1_wild_1345_segnet_averagePool_featDirection_res18_Z4': ['t_fit',
                                                                 'model/m_segnet_averagePool_featDirection',
                                                                 'wild_configs/d_wildtrack_1345_Z4', [0, 2, 4, 6],
                                                                 'auxiliary/no_auxiliary'],
    'fit1_wild_0246_segnet_averagePool_featDirection_res18_Z4': ['t_fit',
                                                                 'model/m_segnet_averagePool_featDirection',
                                                                 'wild_configs/d_wildtrack_0246_Z4', [1, 3, 4, 5],
                                                                 'auxiliary/no_auxiliary'],

    'fit1_wild_1345_segnet_averagePool_original_featDirection_res18_Z4': ['t_fit',
                                                                          'model/m_segnet_averagePool_featDirection',
                                                                          'wild_configs/d_wildtrack_1345_Z4',
                                                                          [0, 2, 4, 6],
                                                                          'auxiliary/use_all_pixels_feat_direction'],
    'fit1_wild_0246_segnet_averagePool_original_featDirection_res18_Z4': ['t_fit',
                                                                          'model/m_segnet_averagePool_featDirection',
                                                                          'wild_configs/d_wildtrack_0246_Z4',
                                                                          [1, 3, 4, 5],
                                                                          'auxiliary/use_all_pixels_feat_direction'],

}

tests_129_1 = {
    'fit2_wild_1345_segnet_averagePool_res18_Z4': ['t_fit_BS2',
                                                   'model/m_segnet_averagePool',
                                                   'wild_configs/d_wildtrack_1345_Z4', [0, 2, 4, 6],
                                                   'auxiliary/no_auxiliary'],
    'fit2_wild_0246_segnet_averagePool_res18_Z4': ['t_fit_BS2',
                                                   'model/m_segnet_averagePool',
                                                   'wild_configs/d_wildtrack_0246_Z4', [1, 3, 4, 5],
                                                   'auxiliary/no_auxiliary'],

    'fit2_wild_1345_segnet_averagePool_featDirection_res18_Z4': ['t_fit_BS2',
                                                                 'model/m_segnet_averagePool_featDirection',
                                                                 'wild_configs/d_wildtrack_1345_Z4', [0, 2, 4, 6],
                                                                 'auxiliary/no_auxiliary'],
    'fit2_wild_0246_segnet_averagePool_featDirection_res18_Z4': ['t_fit_BS2',
                                                                 'model/m_segnet_averagePool_featDirection',
                                                                 'wild_configs/d_wildtrack_0246_Z4', [1, 3, 4, 5],
                                                                 'auxiliary/no_auxiliary'],

    'fit2_wild_1345_segnet_averagePool_original_featDirection_res18_Z4': ['t_fit_BS2',
                                                                          'model/m_segnet_averagePool_featDirection',
                                                                          'wild_configs/d_wildtrack_1345_Z4',
                                                                          [0, 2, 4, 6],
                                                                          'auxiliary/use_all_pixels_feat_direction'],
    'fit2_wild_0246_segnet_averagePool_original_featDirection_res18_Z4': ['t_fit_BS2',
                                                                          'model/m_segnet_averagePool_featDirection',
                                                                          'wild_configs/d_wildtrack_0246_Z4',
                                                                          [1, 3, 4, 5],
                                                                          'auxiliary/use_all_pixels_feat_direction'],

    # 'fit2_wild_1345_mvdet': ['t_fit_2', 'model/m_mvdet', 'wild_configs/d_wildtrack_1345_Z4',
    #                          [0, 2, 4, 6], 'auxiliary/no_auxiliary'],
    # 'fit2_wild_0246_mvdet': ['t_fit_2', 'model/m_mvdet', 'wild_configs/d_wildtrack_0246_Z4',
    #                          [1, 3, 4, 5], 'auxiliary/no_auxiliary'],
}

tests_129_2 = {

    'fit2_wild_1345_segnet_512_original_featDirection_res18_Z4': ['t_fit_2',
                                                                  'model/m_segnet_CameraCountWA_featDirection',
                                                                  'wild_configs/d_wildtrack_1345_Z4',
                                                                  [0, 2, 4, 6],
                                                                  'auxiliary/use_all_pixels_feat_direction'],

    'fit2_wild_0246_segnet_512_original_featDirection_res18_Z4': ['t_fit_2',
                                                                  'model/m_segnet_CameraCountWA_featDirection',
                                                                  'wild_configs/d_wildtrack_0246_Z4',
                                                                  [1, 3, 4, 5],
                                                                  'auxiliary/use_all_pixels_feat_direction'],
}

tests_129_mean = {

    'fit2_wild_1345_segnet_512_original_featDirection_res18_Z4': ['t_fit_2',
                                                                  'model/m_segnet_mean',
                                                                  'wild_configs/d_wildtrack_1345_Z4',
                                                                  [0, 2, 4, 6],
                                                                  'auxiliary/no_auxiliary'],

    'fit2_wild_0246_segnet_512_original_featDirection_res18_Z4': ['t_fit_2',
                                                                  'model/m_segnet_mean',
                                                                  'wild_configs/d_wildtrack_0246_Z4',
                                                                  [1, 3, 4, 5],
                                                                  'auxiliary/no_auxiliary'],
}

tests_129_averagePool = {

    'fit2_wild_1345_segnet_512_original_featDirection_res18_Z4': ['t_fit_2',
                                                                  'model/m_segnet_averagePool',
                                                                  'wild_configs/d_wildtrack_1345_Z4',
                                                                  [0, 2, 4, 6],
                                                                  'auxiliary/no_auxiliary'],

    'fit2_wild_0246_segnet_512_original_featDirection_res18_Z4': ['t_fit_2',
                                                                  'model/m_segnet_averagePool',
                                                                  'wild_configs/d_wildtrack_0246_Z4',
                                                                  [1, 3, 4, 5],
                                                                  'auxiliary/no_auxiliary'],
}

### IPI8
# tests_8 = {
#     'fit2_wild_all_segnet_averagePool_512_res18_Z4': ['t_fit_IPI8', 'model/m_segnet_averagePool',
#                                                       'wild_configs/d_wildtrack_IPI8',
#                                                       [0, 1, 2, 3, 4, 5, 6],
#                                                       'auxiliary/no_auxiliary'],
# }

tests_8 = {
    'fit2_wild_all_segnet_averagePool_512_res18_Z4': ['t_fit_IPI8', 
                                                      'model/m_segnet_maxPool', # mean or average pool
                                                      'wild_configs/d_wildtrack_IPI8',
                                                      [0, 1, 2, 3, 4, 5, 6],
                                                      'auxiliary/no_auxiliary'
                                                      ]
}

tests_8_1 = {
    'fit2_wild_all_segnet_averagePool_512_featDirection_res18_Z4': ['t_fit_IPI8_1',
                                                                    'model/m_segnet_averagePool_featDirection',
                                                                    'wild_configs/d_wildtrack_IPI8',
                                                                    [0, 1, 2, 3, 4, 5, 6],
                                                                    'auxiliary/no_auxiliary'],

}

tests_8_2 = {
    'fit2_wild_all_segnet_averagePool_512_original_featDirection_res18_Z4': ['t_fit_IPI8_2',
                                                                             'model/m_segnet_averagePool_featDirection',
                                                                             'wild_configs/d_wildtrack_IPI8',
                                                                             [0, 1, 2, 3, 4, 5, 6],
                                                                             'auxiliary/use_all_pixels_feat_direction'],

}
