## 项目内容
本项目主要针对文本分类场景，尝试目前流行的深度学习模型，实现文本分类的目的。  
主要模型：
- Text Classification with CNN
- Text Classification with RNN

------
## Text Classification with CNN
使用基于tensorflow的卷积神经网络进行中文文本分类。  
本项目借鉴：https://github.com/gaussic/text-classification-cnn-rnn  

### 环境

| 模块 | 版本 |
| :----------: | :----------: |
| python | 2.7.13 |
| tensorflow | 1.10.0 |
| numpy | 1.16.2 |
| scikit-learn | 0.19.0 |
| scipy | 0.19.1 |

### 数据集
使用THUCNews的一个子集进行训练与测试，数据集请自行到[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)下载，请遵循数据提供方的开源协议。  
本项目使用THUCNews数据集所有类别共计14个，每个类别10000条数据。下载链接: [https://pan.baidu.com/s/1lG2MgbPznSHiuiqsn0ZWvg](https://pan.baidu.com/s/1lG2MgbPznSHiuiqsn0ZWvg)，密码：8l0s     
类别如下：  

```
体育, 娱乐, 家居, 彩票, 房产, 教育, 时尚, 时政, 星座, 游戏, 社会, 科技, 股票, 财经
```

数据集划分如下：  

| 训练集 | 验证集 | 测试集 |
| :----------: | :----------: | :----------: |
| 8000*14 | 1000*14 | 1000*14 |

从原始数据集生成子集的过程参考`helper`下的两个脚本。其中，`copy_data.py`用于从每个分类拷贝10000个文件，`cnews_group.py`用于将多个文件整合到一个文件中。执行`cnews_loader.py`文件后，得到三个数据文件。下载链接: [https://pan.baidu.com/s/1mCZkDsdiImJSmmu57g7mfg](https://pan.baidu.com/s/1mCZkDsdiImJSmmu57g7mfg)，密码：ogiw  

| 文件 | 类型 | 数目 |
| :----------: | :----------: | :----------: |
| cnews.train.txt | 训练集 | 112000 |
| cnews.val.txt | 验证集 | 14000 |
| cnews.test.txt | 测试集 | 14000 |

### 预处理

`data/cnews_loader.py`为数据的预处理文件。

- `read_file()`: 读取文件数据;
- `build_vocab()`: 构建词汇表，使用字符级的表示，这一函数会将词汇表存储下来，避免每一次重复处理;
- `read_vocab()`: 读取上一步存储的词汇表，转换为`{词：id}`表示;
- `read_category()`: 将分类目录固定，转换为`{类别: id}`表示;
- `to_words()`: 将一条由id表示的数据重新转换为文字;
- `process_file()`: 将数据集从文字转换为固定长度的id序列表示;
- `batch_iter()`: 为神经网络的训练准备经过shuffle的批次的数据。

经过数据预处理，数据的格式如下：

| Data | Shape | Data | Shape |
| :---------- | :---------- | :---------- | :---------- |
| x_train | [112000, 600] | y_train | [112000, 14] |
| x_val | [14000, 600] | y_val | [14000, 14] |
| x_test | [14000, 600] | y_test | [14000, 14] |






