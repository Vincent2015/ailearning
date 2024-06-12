0,机器学习项目的5步: 定义问题（营销/运营/维护等）—>收集数据（A/B测试）和预处理/-->（选择算法和确定模型—>训练模型--->评估并优化模型)，很多时候，后面3步需要不断的反复循环的，甚至有时候后面四步 包含数据的处理特别是涉及特征工程话，也需要反复。在机器学习数据集方面，针对复杂数据集，可在原数据集外通过pandas 和numpy单独构建训练集和测试集（例如RMF数据集的构建）

1.kaggle 初次使用vgg19 会因为网络原因导致下载不下来，需要在 setting页面打开 internet选项

2. 普通的 CNN，和VGG19使用的Y的特征数量不同。在迁移学习章节中，要注意。https://blog.csdn.net/qq_39938666/article/details/93424418。

可以通过,更改Y的特征数量

num_classes = 10 && y_labels = to_categorical(_ylabels, num_classes=num_classes)

3.optimize.Adma不可用时，用tf.keras.optimize.Adma代替

4.使用模型预测传入的变量个数要和模型训练时使用的X的特征个数一致，否则会导致线上应用调用不通过------见于线性回归。

5.性能:VGG 的> 数据增强的普通模型>普通模型

6.（Dense层为主）DNN iput_dim=17 的input.shape结果为（None,17）,其units数量按经验为 12，24等。

7.CNN的因为是局部连接，因此参数大大少于Dense全连接网络

8,神队学习准确率提升：数据增强，Dropout(防过拟合)，更新优化器并设置学习率

9.用户留存，Kaplan生存模型及COX危害系数可以预测具体一个人的流失风险

10.对于分类模型中的样本不均衡问题，需要结合混淆矩阵，F1分数，和AUC曲线一起看模型的优劣，不能只看准确率。

11.seaborn 绘图时，sns.countplot(是否转化',data=df_fission)这个调用会报错**countplot() got multiple values for argument 'data’****，需要给第一个参数加上变量名如下**

sns.countplot(x='是否转化',data=df_fission)

12.sklearn的线性回归 算法LinearRegression 的 coef_ 和 intercept_ 属性分别包含各个特征的权重和模型的偏置，它们是模型的内部参数。除了线性回归，常用的回归算法有

贝叶斯回归(使用贝叶斯推理的线性回归)，SVM回归（分为核SVM和？svm）, 决策树（CART classification and regression），随机森林，AdaBoost和XGBOoost等Boosting 方法的集成算法（梯度提升），神经网络DNN —等等

13,通过特征选择（除了数据分析观察，可以通过sklearn的SelectBest自动选择特征的方法来自动选择特征，除了SelectBest,还有RFE工具，SelectFromModel工具，SequentialFeature selector工具等。最后还可以通过PCA,LDA的降维算法进行数据特征选择），特征变化【连续特征主要改变分布或者压缩特征空间，在sklearn中常用的方法有StandardScaler,MinMaxScaler,RobusterScaler,Normalizer-规范化缩放；其次还有针对类别特证的变化：虚拟变量（当特证有 m个不同类别时，get_dummies将得到m-1二进制特征）和one-Hot编码，可以通过Pandas的方法get_dummies进行转换；还有针对数值型特征的离散化处理方法—**分桶**，主要针对特征数量级跨度较大的特征】，特征构建（创新特征如RMF）的三个基本思路，来提升模型的效率.

--补充：特征选择，特征相关性分析也是一个重要的方法。

\14. 防止过拟合的方法，除了增加数据量，特征工程及选择简单的模型（缇卡姆法则）针对决策树的剪枝方法及线性回归的正则化方法（L1正则化的Lasso回归和L2正则化的Ridge岭回归）

15.数据集越大，就越不容易出现过拟合的现象。那么，如何利用较小的数据集，从而达到较大数据集的效果呢？这就需要交叉验证。交叉验证虽然一直在用不同的数据拆分进行模型的拟合，但它实际上并不是在试图训练出新的模型，它只是我们对模型的一种评估方式而已。

16，网格搜索sk-learn 中有一个 GridSearchCV 工具，中文叫做网格搜索，堪称是辅助我们自动调参的神器，它可以帮我们自动调参，轻松找到模型的最优参数。GridSearchCV 会在后台创建出一大堆的并行进程，挨个执行各种超参数的组合，同时还会使用交叉验证的方法（名称中的 CV，意思就是 cross validation），来评估每个超参数组合的模型。最后，GridSearchCV 会帮你选定哪个组合是给定模型的最佳超参数值.

model_rfr = RandomForestClassifier() # 随机森林模型

\# 对随机森林算法进行参数优化

rf_param_grid = {"max_depth": [None],

​         "max_features": [3, 5, 12],

​         "min_samples_split": [2, 5, 10],

​         "min_samples_leaf": [3, 5, 10],

​         "bootstrap": [False],

​         "n_estimators" :[100,300],

​         "criterion": ["gini"]}

from sklearn.model_selection import GridSearchCV # 导入网格搜索工具

model_rfr_gs = GridSearchCV(model_rfr,

​              param_grid = rfr_param_grid, cv=3,

​              scoring="r2", n_jobs= 10, verbose = 1)

model_rfr_gs.fit(X_train, y_train) # 用优化后的参数拟合训练数据集



经过 GridSearchCV 自动地换参、拟合并自动交叉验证评估后，最佳参数组合实际上已经被选出了，它就被存储在 model_rfr_gs 这个新的随机森林中，我们可以直接用它来做预测。这里，我们调用 model_rfr_gs 的 best_params_ 属性，可以查看一下这个最优模型是由哪些超参数组合而成的。

16.DNN,CNN,RNN的网络结构及参数差异：看一个神经网络是普通神经网络 DNN，还是 CNN 或者 RNN 呢？这其中的关键就是看输入层和中间层主要是什么类型。DNN 的输入层和中间层主要是 Dense 层，CNN 的输入层和中间层主要是 Conv1D、Conv2D 或者 Conv3D，RNN 的输入层和中间层主要是 SimpleRNN 或者 GRU 或者 LSTM 层。其次还有池化层，dropout层等概念及作用的理解。

17.机器学习算法的选择

关于快速定位合适的机器学习算法，这其中的要点是定义好问题，明确是监督学习问题还是无监督学习问题，是分类问题还是回归问题，这能让我们排除掉不相干的算法。在选择具体的算法时，建议从训练数据的大小、特征的数量、是着重考量模型的性能还是考量模型的可解释性、是否要求模型有很快的训练速度，以及数据的线性程度这几个方面，来选择最适宜的算法。另外，建议从探索数据开始，熟悉你的数据。因为“更好的数据往往胜过更好的算法”，对数据集的特征之间的性质了解得越清楚，越有可能得到高性能的模型。当然，对模型背后的原理和假设越了解，以及模型外部参数设定的含义越熟悉，也越有可能得到高性能的模型。在定位合适的机器学习算法时，还有一个最基本的原则是，从简单的模型开始构建基准模型，然后尝试更复杂的方法。最后，请始终记住，尽可能地尝试多种算法，多种参数组合，然后比较它们的性能，以选择最适合特定任务的算法。但也不要忘记尝试集成方法，因为它通过博彩众长，能提供更好的准确性

