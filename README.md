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

## CNN卷积神经网络

### 配置项

CNN可配置的参数如下所示，在`cnn_model.py`中。

```python
class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64      # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 10        # 类别数
    num_filters = 128        # 卷积核数目
    kernel_size = 5         # 卷积核尺寸
    vocab_size = 5000       # 词汇表达小

    hidden_dim = 128        # 全连接层神经元

    dropout_keep_prob = 0.5 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 64         # 每批训练大小
    num_epochs = 10         # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard
```

### CNN模型

具体参看`cnn_model.py`的实现。

大致结构如下：

![images/cnn_architecture](images/cnn_architecture.png)

### 训练与验证

运行 `python run_cnn.py train`，可以开始训练。

> 若之前进行过训练，请把tensorboard/textcnn删除，避免TensorBoard多次训练结果重叠。

```
Epoch: 1
Iter:      0, Train Loss:    2.3, Train Acc:  15.62%, Val Loss:    2.3, Val Acc:  10.16%, Time: 0:00:01 *
Iter:    100, Train Loss:   0.63, Train Acc:  79.69%, Val Loss:    1.1, Val Acc:  71.00%, Time: 0:00:03 *
Iter:    200, Train Loss:   0.48, Train Acc:  84.38%, Val Loss:   0.67, Val Acc:  80.10%, Time: 0:00:05 *
Iter:    300, Train Loss:   0.32, Train Acc:  92.19%, Val Loss:   0.48, Val Acc:  84.72%, Time: 0:00:07 *
Iter:    400, Train Loss:   0.12, Train Acc:  98.44%, Val Loss:   0.31, Val Acc:  91.60%, Time: 0:00:08 *
Iter:    500, Train Loss:   0.18, Train Acc:  98.44%, Val Loss:   0.29, Val Acc:  91.42%, Time: 0:00:10 
Iter:    600, Train Loss:   0.22, Train Acc:  95.31%, Val Loss:   0.29, Val Acc:  92.24%, Time: 0:00:12 *
Iter:    700, Train Loss:   0.15, Train Acc:  93.75%, Val Loss:   0.24, Val Acc:  93.50%, Time: 0:00:13 *
Epoch: 2
Iter:    800, Train Loss:  0.078, Train Acc:  96.88%, Val Loss:   0.23, Val Acc:  93.82%, Time: 0:00:15 *
Iter:    900, Train Loss:  0.088, Train Acc:  96.88%, Val Loss:   0.18, Val Acc:  95.00%, Time: 0:00:17 *
Iter:   1000, Train Loss:  0.079, Train Acc:  98.44%, Val Loss:   0.27, Val Acc:  92.18%, Time: 0:00:18 
Iter:   1100, Train Loss:   0.13, Train Acc:  96.88%, Val Loss:   0.23, Val Acc:  93.76%, Time: 0:00:20 
Iter:   1200, Train Loss:    0.2, Train Acc:  93.75%, Val Loss:   0.18, Val Acc:  95.14%, Time: 0:00:22 *
Iter:   1300, Train Loss:  0.074, Train Acc:  96.88%, Val Loss:   0.18, Val Acc:  95.58%, Time: 0:00:23 *
Iter:   1400, Train Loss:  0.067, Train Acc:  96.88%, Val Loss:   0.19, Val Acc:  94.58%, Time: 0:00:25 
Iter:   1500, Train Loss:   0.02, Train Acc: 100.00%, Val Loss:    0.2, Val Acc:  94.32%, Time: 0:00:27 
Epoch: 3
Iter:   1600, Train Loss:   0.03, Train Acc: 100.00%, Val Loss:   0.18, Val Acc:  95.18%, Time: 0:00:28 
Iter:   1700, Train Loss:  0.099, Train Acc:  96.88%, Val Loss:   0.18, Val Acc:  95.02%, Time: 0:00:30 
Iter:   1800, Train Loss:  0.052, Train Acc:  96.88%, Val Loss:   0.19, Val Acc:  94.88%, Time: 0:00:32 
Iter:   1900, Train Loss:  0.043, Train Acc:  98.44%, Val Loss:   0.19, Val Acc:  95.00%, Time: 0:00:33 
Iter:   2000, Train Loss:  0.082, Train Acc:  96.88%, Val Loss:    0.2, Val Acc:  94.70%, Time: 0:00:35 
Iter:   2100, Train Loss:  0.088, Train Acc:  98.44%, Val Loss:   0.18, Val Acc:  94.96%, Time: 0:00:36 
Iter:   2200, Train Loss:  0.017, Train Acc: 100.00%, Val Loss:   0.19, Val Acc:  94.86%, Time: 0:00:38 
Iter:   2300, Train Loss:  0.059, Train Acc:  98.44%, Val Loss:   0.18, Val Acc:  95.36%, Time: 0:00:40 
No optimization for a long time, auto-stopping...
```

在验证集上的最佳效果为95.36%，且只经过了3轮迭代就已经停止。

准确率和误差如图所示：

![images](images/acc_loss.png)


### 测试

运行 `python run_cnn.py test` 在测试集上进行测试。

```
Test Loss:   0.12, Test Acc:  96.60%
Precision, Recall and F1-Score...
             precision    recall  f1-score   support

         体育       0.99      0.99      0.99      1000
         财经       0.95      0.99      0.97      1000
         房产       1.00      0.99      1.00      1000
         家居       0.97      0.90      0.93      1000
         教育       0.94      0.95      0.94      1000
         科技       0.97      0.97      0.97      1000
         时尚       0.94      0.98      0.96      1000
         时政       0.96      0.94      0.95      1000
         游戏       0.98      0.97      0.98      1000
         娱乐       0.97      0.97      0.97      1000

avg / total       0.97      0.97      0.97     10000


Confusion Matrix...
[[992   0   0   1   5   1   0   1   0   0]
 [  0 995   0   0   0   1   0   4   0   0]
 [  1   1 995   1   2   0   0   0   0   0]
 [  2  19   1 904  18   8  17  22   3   6]
 [  2   7   0   6 946   6  12  10   5   6]
 [  0   3   0   5   3 967  14   1   6   1]
 [  1   0   0   6   5   4 977   0   0   7]
 [  0  21   1   6  20   9   0 937   1   5]
 [  1   3   0   1   4   2  12   0 972   5]
 [  1   0   0   5   8   3   6   0   2 975]]

```

在测试集上的准确率达到了96.6%，且各类的precision, recall和f1-score都超过了0.9。

从混淆矩阵也可以看出分类效果非常优秀。
