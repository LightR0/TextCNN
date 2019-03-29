# 项目内容
本项目主要针对文本分类场景，尝试目前流行的深度学习模型，实现文本分类的目的。  
主要模型：
- Text Classification with CNN
- Text Classification with RNN

------
# Text Classification with CNN
使用基于tensorflow的卷积神经网络进行中文文本分类。  
本项目引用借鉴：https://github.com/gaussic/text-classification-cnn-rnn  

## 环境

| 模块 | 版本 |
| :----------: | :----------: |
| python | 2.7.13 |
| tensorflow | 1.10.0 |
| numpy | 1.16.2 |
| scikit-learn | 0.19.0 |
| scipy | 0.19.1 |

## 数据集
使用THUCNews的一个子集进行训练与测试，数据集请自行到[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)下载，请遵循数据提供方的开源协议。  
本项目使用THUCNews数据集所有类别共计10个，每个类别6500条数据。    
类别如下：  

```
体育, 娱乐, 家居, 教育, 时政, 游戏, 社会, 科技, 股票, 财经
```

数据集划分如下：  

| 训练集 | 验证集 | 测试集 |
| :----------: | :----------: | :----------: |
| 5000*10 | 500*10 | 1000*10 |

从原始数据集生成子集的过程参考`helper`下的两个脚本。其中，`copy_data.py`用于从每个分类拷贝6500个文件，`cnews_group.py`用于将多个文件整合到一个文件中。执行`cnews_loader.py`文件后，得到三个数据文件。   
链接: https://pan.baidu.com/s/1hugrfRu 密码: qfud

| 文件 | 类型 | 数目 |
| :----------: | :----------: | :----------: |
| cnews.train.txt | 训练集 | 50000 |
| cnews.val.txt | 验证集 | 5000 |
| cnews.test.txt | 测试集 | 10000 |

## 预处理

`data/cnews_loader.py`为数据的预处理文件。

| 函数        | 说明        |  
| :---------- | :---------- |
| read_file() | 读取文件数据 |
| build_vocab() | 构建词汇表，使用字符级的表示，这一函数会将词汇表存储下来，避免每一次重复处理 |
| read_vocab()  | 读取上一步存储的词汇表，转换为`{词：id}`表示 |
| read_category() | 将分类目录固定，转换为`{类别: id}`表示 |
| to_words() | 将一条由id表示的数据重新转换为文字 |
| process_file() | 将数据集从文字转换为固定长度的id序列表示 |
| batch_iter() | 为神经网络的训练准备经过shuffle的批次的数据 |

经过数据预处理，数据的格式如下：

| Data | Shape | Data | Shape |
| :---------- | :---------- | :---------- | :---------- |
| x_train | [50000, 600] | y_train | [50000, 10] |
| x_val | [5000, 600] | y_val | [5000, 10] |
| x_test | [10000, 600] | y_test | [10000, 10] |






