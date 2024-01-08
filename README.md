# GCE-GNN-PytorchGeometric

论文Global Context Enhanced Graph Neural Networks for Session-based Recommendation的代码复现

## 运行

在dataset目录下执行python build_global_graph.py --dataset=Tmall --epsilon=2

在根目录下执行python main.py --dataset=Tmall
(也许需要大概2个小时以上在Apple M2)
(没有对cuda适配，有需要自行添加额外代码)

## 查看运行结果

tensorboard --logdir=log

