import matplotlib.pyplot as plt
import pandas as pd

COLUMN = ['dataset', 'views', 'mode', 'version', 'backbone',
          'extra_1', 'extra_2', 'extra_3', 'test_views',
          'test_on_diff_views', 'num_views']

FILE_PATH = 'util_outputs/simplified_df.xlsx'


def get_all_tests(df):
    tests_list = df['Test Name']
    tests_names = [split.split('_') for split in tests_list]

    # tests_names_2 = [split if split[3].isdigit() else split.insert(3, '1') for split in tests_names]
    refind_tests_names = []
    for test in tests_names:
        # first 3 elements are always ['dataset', 'views', 'mode']
        # element 3: if int --> version; else: insert '1'
        # element 4: backbone
        # ensure there are len(COLUMN) fill the others with ''
        # element 5 if not .isdigit(): extra_1 test_view=views; else: test_views
        refind_test = []
        for i, split in enumerate(test):
            if i in [0, 1, 2]:
                refind_test.append(test[i])
            elif i == 3:
                if not test[3].isdigit():
                    test.insert(3, '1')
                    refind_test.append('1')
                else:
                    refind_test.append(test[3])

            elif i == 4:
                refind_test.append(test[4])

            elif i == 5:
                if test[5] == '512':
                    refind_test.append(test[5])
                else:
                    if test[5].isdigit():  # is a test
                        refind_test.append('')
                        refind_test.append('')
                        refind_test.append('')
                        refind_test.append(test[5])
                        refind_test.append(True)
                    else:  # extra 2 (Z8)
                        refind_test.append('')
                        refind_test.append(test[5])
                        refind_test.append('')

            elif i == 6:  # is a test
                while len(refind_test) < 8:  # we are at the end --> ensure to have test at the end
                    refind_test.append('')
                refind_test.append(test[6])
                refind_test.append(True)

        # print(len(COLUMN), len(refind_test))
        while len(refind_test) < len(COLUMN) - 1:
            refind_test.append('')
        refind_test.append(len(test[1]))
        refind_tests_names.append(refind_test)
        # print('\n')
        # print(test)
        # print(refind_test)
    return refind_tests_names


def test_name_condition(df, conditions):
    data_list = get_all_tests(df)

    # Convert data_list to a pandas DataFrame
    df = pd.DataFrame(data_list, columns=COLUMN)

    # Apply conditions to filter the DataFrame
    filtered_df = df.copy()
    for key, value in conditions.items():
        filtered_df = filtered_df[filtered_df[key] == value]

    # Convert filtered DataFrame back to a list of lists
    filtered_data_list = filtered_df.values.tolist()

    results_to_compare = ['_'.join([str(s) for s in sublist if s != ''][:-2]) for sublist in filtered_data_list]

    return results_to_compare

def get_tests_test_on_diff_views(df,test_name):
    tests_list = df['Test Name']
    filtered_names = list(filter(lambda name: name.lower().startswith(test_name), tests_list))
    return filtered_names

if __name__ == "__main__":
    # Read the Excel file
    df = pd.read_excel(FILE_PATH)
    df = df.dropna()
    data_list = get_all_tests(df)

    conditions = {
        'dataset': 'wild',
        #'mode': 'mvdet',
        #'version': '1',
        'num_views': 2,
        'backbone': 'res18',
    }
    results_to_compare = test_name_condition(df, conditions)
    test_name = get_tests_test_on_diff_views(df, 'wild_25_mvdet_2')
    print(results_to_compare)
    print(test_name)

    # Output: [['dataset1', 100, 'mode1', 'version1', 'test1']]
