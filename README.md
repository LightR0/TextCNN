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
本项目使用THUCNews数据集所有类别共计14个，每个类别10000条数据。  
类别如下：  
```
体育, 娱乐, 家居, 彩票, 房产, 教育, 时尚, 时政, 星座, 游戏, 社会, 科技, 股票, 财经
```
这个子集可以在此下载，链接：[http://thuctc.thunlp.org/](http://thuctc.thunlp.org/)，密码：  
数据集划分如下：  

| 训练集 | 验证集 | 测试集 |
| :----------: | :----------: | :----------: |
| 8000*14 | 1000*14 | 1000*14 |

从原始数据集生成子集的过程参考`helper`下的两个脚本。其中，`copy_data.py`用于从每个分类拷贝10000个文件，`cnews_group.py`用于将多个文件整合到一个文件中。执行`cnews_loader.py`文件后，得到三个数据文件：  

| 文件 | 类型 | 数目 |
| :----------: | :----------: | :----------: |
| cnews.train.txt | 训练集 | 112000 |
| cnews.val.txt | 验证集 | 14000 |
| cnews.test.txt | 测试集 | 14000 |

这三个文件可以在此下载，链接：[http://thuctc.thunlp.org/](http://thuctc.thunlp.org/)，密码：  




