import os
import numpy as np


def read_data(root_dir):
    temp_dir = root_dir
    # 遍历文件夹中的所有文件
    file_list = [filename for filename in os.listdir(temp_dir) if filename.endswith(".npy")]
    # selected_files = random.sample(file_list, 200)
    X_sampled = []
    Y_sampled = []

    for filename in file_list:
        # 加载.npy文件的内容
        data = np.load(os.path.join(temp_dir, filename))

        # 提取文件名中的数字
        parts = filename.split("_")
        if len(parts) == 2:
            # 获取第二部分中的数字并转换为整数
            num = int(parts[1].split(".")[0])
            X_sampled.append(data)
            Y_sampled.append(num)

    X_sampled = np.array(X_sampled).squeeze()
    # Y_sampled = np.array(Y_sampled).reshape(X_sampled.shape[0], 1)
    Y_sampled = np.array(Y_sampled)

    return X_sampled, Y_sampled

