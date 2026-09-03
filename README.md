# Python-Mini-Project-Speech-Emotion-Recognition-with-librosa


## 模型设置

- 分类器：MLPClassifier（多层感知机）
- 隐藏层 (300,)，alpha=0.01，batch_size=256，max_iter=500
- 训练集 / 测试集 = 75% / 25%

## 结果

在测试集上取得约 **59% 的准确率**（四分类任务，随机猜测的基准为 25%）。

> 备注：准确率比原教程（72.4%）略低，可能与我使用的是完整版立体声数据、
> 并做了单声道转换有关；模型本身也存在随机性。

## 致谢

本项目基于 [DataFlair 的语音情绪识别教程](https://data-flair.training/blogs/python-mini-project-speech-emotion-recognition/)
学习完成，代码思路与整体流程来自该开源教程，**非本人原创**。

## Model Configuration

- Classifier: MLPClassifier
- Hidden layer (300,), alpha=0.01, batch_size=256, max_iter=500
- Train / test split = 75% / 25%

## Results

Achieved about **59% accuracy** on the test set (4-class classification;
the random-guess baseline is 25%).

> Note: The accuracy is slightly lower than the original tutorial's (72.4%),
> likely because I used the full stereo dataset and converted it to mono;
> the model itself also has inherent randomness.

## Acknowledgements

This project was built by following the
[DataFlair Speech Emotion Recognition tutorial](https://data-flair.training/blogs/python-mini-project-speech-emotion-recognition/).
The code logic and overall workflow come from that open-source tutorial —
**this is not my original work**.