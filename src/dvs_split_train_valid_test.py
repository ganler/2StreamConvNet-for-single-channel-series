import os
import pandas as pd

'''
假设我们的代码格式都是这样：
- Project
    - src
        codes
    - DataSet
        - class_0
        - class_1
        ...
        - class_N

假设我们的数据格式是这样：
-DataSet
    - class_0
        - sample_0
            - spatial
                - xxx.png
                ...
            - motion
                - xxx.png
                ...
        - sample_1
        ...
    - class_1
        ...
    ...
    - class_N
本.py最后生成3个csv文件，分别是
- train.csv,
- valid.csv
- test.csv
'''

# ==========================================
_dict = {'s_frame_0':[], 's_frame_1':[], 's_frame_2':[], 'm_frame_0':[], 'm_frame_1':[], 'm_frame_2':[], 'tags': []}

dataset_loc = '../DataSet/'

# csv的路径
train_csv = 'train.csv'
valid_csv = 'valid.csv'
test_csv = 'test.csv'

split_pro = [0.7, 0.15, 0.15]
# ==========================================

def csv_generate(split, shuffle=True):
    for root, s_dirs, _ in os.walk(dataset_loc, topdown=True):
        for sub_dir in s_dirs:
            if str(sub_dir)[0] != 'c':
                continue
            tag = str(sub_dir).split('_')[1]        # tag = class_N中的N
            class_dir = os.path.join(root, sub_dir) # class_dir 是class_x的folder的位置
            video_dir_list = os.listdir(class_dir)  # 其中的sample
            for i in range(len(video_dir_list)):
                spatial_sample_list = os.listdir(str(class_dir)+'/'+str(video_dir_list[i])+'/spatial')
                motion_sample_list = os.listdir(str(class_dir)+'/'+str(video_dir_list[i])+'/motion')
                if len(spatial_sample_list) == 0:
                    print(str(class_dir) + '/' + str(video_dir_list[i]) + '/spatial -> is empty')
                if len(motion_sample_list) == 0:
                    print(str(class_dir) + '/' + str(video_dir_list[i]) + '/motion -> is empty')
                for j in range(len(motion_sample_list)-2):
                    if spatial_sample_list:
                        _dict['s_frame_0'].append(
                            str(str(class_dir) + '/' + str(video_dir_list[i]) + '/spatial/') + str(spatial_sample_list[j]))
                        _dict['s_frame_1'].append(
                            str(str(class_dir) + '/' + str(video_dir_list[i]) + '/spatial/') + str(spatial_sample_list[j+1]))
                        _dict['s_frame_2'].append(
                            str(str(class_dir) + '/' + str(video_dir_list[i]) + '/spatial/') + str(spatial_sample_list[j+2]))
                    else:
                        _dict['s_frame_0'].append('null')
                        _dict['s_frame_1'].append('null')
                        _dict['s_frame_2'].append('null')
                    if motion_sample_list:
                        _dict['m_frame_0'].append(
                            str(str(class_dir) + '/' + str(video_dir_list[i]) + '/motion/') + str(motion_sample_list[j]))
                        _dict['m_frame_1'].append(
                            str(str(class_dir) + '/' + str(video_dir_list[i]) + '/motion/') + str(motion_sample_list[j+1]))
                        _dict['m_frame_2'].append(
                            str(str(class_dir) + '/' + str(video_dir_list[i]) + '/motion/') + str(motion_sample_list[j+2]))
                    else:
                        _dict['m_frame_0'].append('null')
                        _dict['m_frame_1'].append('null')
                        _dict['m_frame_2'].append('null')
                    _dict['tags'].append(tag)

    # 打乱并切分
    df = pd.DataFrame(_dict).sample(frac=shuffle)

    if shuffle:
        df = df.reset_index()
    train = df.loc[0 : int(split[0]*len(df))]

    valid = df.loc[int(split[0]*len(df)) : int(len(df)*(split[0]+split[1]))]

    test = df.loc[int((split[0]+split[1])*len(df)) : len(df)-1]

    train.to_csv(train_csv)
    valid.to_csv(valid_csv)
    test.to_csv(test_csv)

if __name__ == '__main__':
    csv_generate(split_pro)