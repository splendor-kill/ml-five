# alphagomoku


### 1. 简介

  如名所示，本项目意欲用[alphago\[1\]][1]所采用的的算法来建造一个下五子棋的AI。</br>
  一为学习，二主要是不想用启发式+搜索，因为获得好的启发式很难，且不自由。</br>
  实际上，已有很多[五子棋AI\[2\]][2]很厉害，相关算法可以参考[这儿\[3\]][3]或[这儿\[4\]][4]。</br>
  目前采用新方法的[一些尝试\[5\]][5]还没有达到期望。


### 2. 流程
  就像[alphago\[1\]][1]，我们也想采用类似的方式：  
  * 1. 用监督学习训练策略网络
  * 2. 在前面的基础上，通过self-play来增强策略网络和训练一个值网络
  * 3. 用MCTS结合它们

为了更快的训练，我们先采用9x9的棋盘，待有成功的经验后再移到更大的棋盘上去。</br>
  对于监督学习，首先是收集数据。</br>
  我们的数据采集自gomocup前几名AI对弈记录。</br>
  你可以在[这儿\[6\]][6]下载，它们是经过对称旋转变换的。</br>
  数据集的每一行前81(9x9)个数是棋盘描述，0表示空，1表示黑子，2表示白子，</br>
  后面的数是表示每个位置被访问的次数和赢的次数。


### 3. 代码
  主要的文件有:
  ```
  main.py          #GUI，提供人机对弈界面和其它一些功能入口(通过按键)
  game.py          #一局游戏，由2个AI和1个棋盘组成，可以有或没有GUI
  board.py         #棋盘状态
  strategy.py      #策略(AI)基类
  strategy_dnn.py  #使用dnn作决策的AI
  dnn*.py          #不同结构的DCNN，从本身运行可训练或强化, 用到tensorflow
  mcts.py          #TODO MCTS，使用NN记录统计信息
  mcts1.py         #单线程MCTS，使用Tree记录统计信息
  dfs.py           #另一个AI，来自[7]
  server.py        #用于和gomocup的其它AI切磋，因为gomocup manager[8]
                   #是个Windows程序，而我们的程序主要跑在Linux上，
                   #所以做了一次转发：
                   #gomocup manager <-> Windows stub[9] <-> server.py
```

### 4. 使用
  为了跑起来，你需要在代码里改些配置，主要是dnn.py里面一些目录位置。</br>
  监督学习: python dnn3.py</br>
  强化学习: python main.py, 再按F4</br>
  参与到gomocup manager: python server.py


### 5. 下一步想做的
 * 1. 迁移到更大的棋盘
 * 2. 有swap2规则的AI


### 6. 参考
  [\[1\]Mastering the game of Go with deep neural networks and tree search][1]</br>
  [\[2\]Gomoku AI][2]</br>
  [\[3\]Gomoku Resources][3]</br>
  [\[4\]Carbon Gomoku][4]</br>
  [\[5\]Convolutional and Recurrent Neural Network for Gomoku][5]</br>
  [\[6\]Gomoku Dataset 9x9][6]</br>
  [\[7\]Gobang game with AI in 900 Lines][7]</br>
  [\[8\]Gomocup Manager][8]</br>
  [\[9\]Windows gomocup stub][9]</br>
  [\[10\]Human-level control through deep reinforcement learning][10]</br>
  [\[11\]MCTS][11]


### 7. 踩过的坑
  - 1. 监督学习所用的数据集会严重影响训练效果。</br>
  之前我们用一个比较简单的启发式AI生成的数据集就连简单的规则都学不会，</br>
  和人博弈时简直就是乱下
  - 2. 用CNN描述棋盘状态，同样一些input features用不同形式喂给CNN，对结果影响不大
  - 3. loss曲线和真正的效果感觉很奇怪  
  - 4. 直接和gomocup的其它AI对弈做强化学习实在是太慢，因为它们搜索一步</br>
  都要很久，感觉self-play才是出路，要么能从很少的对弈中学习


### 8. 当前的问题
  - 1. 当前监督学习得到的策略网络预测正确率只有35%，虽然和人玩还是能让人</br>
  感到一些惊喜，但我想应该还要提高一些
  - 2. self-play是这么做的: 让2个相同的AI(权值相同)对弈，其中一个</br>
  不训练(权值保持不变)不探索，另一个用Actor-Critic做强化学习。</br>
  现在的现象是做强化学习的AI的赢率会越来越低，</br>
  value loss能收敛到很小，policy loss不收敛或基本稳定在一个较大的值上面。</br>
  而偶尔能赢的时候恰好是value loss比较大的时候。</br>
  </br>
  alphago做强化学习时候的minibatches是128， 而我们现在是1，是否因为这个影响巨大？</br>
  [DQN\[10\]][10]用了replay memory，如果一局终了时，把的所有状态带上reward后都放到</br>
  replay memory再采样是否有助于解决这个问题？


#### 欢迎讨论，渴望指导，谢谢!
>email: splendor.kill@gmail.com</br>
QQ: 363599755</br>
微信: splendor_k


[1]: http://airesearch.com/wp-content/uploads/2016/01/deepmind-mastering-go.pdf
[2]: http://gomocup.org/download-gomoku-ai/
[3]: http://www.aiexp.info/gomoku-renju-resources-an-overview.html
[4]: http://mczard.republika.pl/gomoku.en.html
[5]: http://cs231n.stanford.edu/reports2016/109_Report.pdf
[6]: https://pan.baidu.com/s/1eS7LBuq
[7]: https://github.com/skywind3000/gobang
[8]: http://gomocup.org/download-gomocup-manager/
[9]: https://github.com/splendor-kill/MyGomocupStub
[10]: https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
[11]: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

