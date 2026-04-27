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

  对于监督学习，首先是收集数据。</br>
  我们的数据采集自gomocup前几名AI对弈记录。</br>
  你可以在[这儿\[6\]][6]下载，它们是经过对称旋转变换的。</br>
  做了几种不同的数据集，9x9和15x15大小的，主要是csv格式，</br>
  数据的含义参考[\[6\]][6]里的README。</br>


### 3. 代码
  项目的主要目录和文件结构如下：
  ```text
  ├── data/                  # 存放训练数据集的目录
  │   └── alphagomoku/       # 包含不同格式和大小的对局数据压缩包
  ├── src/
  │   └── tentacle/          # 核心代码目录
  │       ├── main.py        # GUI，提供人机对弈界面和其它一些功能入口(通过按键)
  │       ├── game.py        # 一局游戏，由2个AI和1个棋盘组成，可以有或没有GUI
  │       ├── board.py       # 棋盘状态管理
  │       ├── config.py      # 项目配置文件
  │       ├── strategy.py    # 策略(AI)基类
  │       ├── strategy_dnn.py# 使用DNN作决策的AI
  │       ├── dnn*.py        # 不同结构的DCNN，本身运行可进行训练或强化学习（依赖PyTorch）
  │       ├── mcts.py        # MCTS实现，使用神经网络记录统计信息
  │       ├── mcts1.py       # 单线程MCTS，使用树结构记录统计信息
  │       ├── dfs.py         # 另一个基于搜索的AI，来自[7]
  │       ├── server.py      # 用于和gomocup的其它AI切磋。因为gomocup manager[8]
  │       │                  # 是个Windows程序，而我们的程序主要跑在Linux上，
  │       │                  # 所以做了一次转发：gomocup manager <-> Windows stub[9] <-> server.py
  │       ├── data_set.py    # 数据集处理相关
  │       ├── ds_loader.py   # 数据加载器
  │       ├── rl_policy.py   # 强化学习策略相关
  │       └── value_net.py   # 价值网络实现
  ├── pyproject.toml         # uv 项目配置及依赖管理文件
  └── README.md              # 项目说明文档
  ```

### 4. 使用
  本项目使用 `uv` 进行环境和依赖管理。首先安装依赖：
  ```bash
  uv sync
  ```
  
  为了跑起来，你需要在代码里改些配置，主要是在 `src/tentacle/config.py` 里。</br>
  命令行统一入口为 `ml-five`（子命令）：</br>
  * 监督学习: `uv run ml-five supervised`（默认 DCNN3；`--network dnn1|dnn2|pre` 可选；`--resume` 从 checkpoint 续训）</br>
  * 强化学习: `uv run ml-five reinforce`（原 GUI 按 F4；`--no-resume` 强制从监督学习权重起步）</br>
  * 图形界面: `uv run ml-five gui`（或仍可用 `uv run ml-five-main`，等价于 `ml-five gui`）</br>
  仍可直接运行: `uv run python src/tentacle/dnn3.py` 等脚本，行为与原先一致。</br>
  参与到gomocup manager: `uv run ml-five-server` (或 `uv run python src/tentacle/server.py`)


### 5. 下一步想做的
 * 1. 探索新的网络结构，增加特征面
 * 2. 解决或绕开self-play过拟合的问题


### 6. 参考
  [\[1\]Mastering the game of Go with deep neural networks and tree search][1]</br>
  [\[2\]Gomoku AI][2]</br>
  [\[3\]Gomoku Resources][3]</br>
  [\[4\]Carbon Gomoku][4]</br>
  [\[5\]Convolutional and Recurrent Neural Network for Gomoku][5]</br>
  [\[6\]Gomoku Dataset][6]</br>
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
  - 1. 当前监督学习得到的策略网络预测正确率只有40%多，应该还有提升空间。
  - 2. self-play的做法和[alphago\[1\]][1]一样的: 在对手池中挑一个对手，与之对弈，</br>
  对手总采用贪婪策略，己方做探索，满足一定条件后加入对手池。</br>
  看起来很完美，但是通过self-play得到的agent会过拟合于自己的对手，虽然可以打败了所有以前的对手，</br>
  但是实际上变得没有通用性了，仅仅是这些以前的那些对手的克星。</br>
  - 3. 单线程[MCTS\[11\]][11]太慢，不实用，得改成[APV-MCTS\[1\]][1]
  
  
#### 欢迎讨论，渴望指导，谢谢!
>email: splendor.kill@gmail.com</br>
QQ: 363599755</br>
微信: splendor_k


[1]: http://airesearch.com/wp-content/uploads/2016/01/deepmind-mastering-go.pdf
[2]: http://gomocup.org/download-gomoku-ai/
[3]: http://www.aiexp.info/gomoku-renju-resources-an-overview.html
[4]: http://mczard.republika.pl/gomoku.en.html
[5]: http://cs231n.stanford.edu/reports2016/109_Report.pdf
[6]: http://pan.baidu.com/s/1eSolIHc
[7]: https://github.com/skywind3000/gobang
[8]: http://gomocup.org/download-gomocup-manager/
[9]: https://github.com/splendor-kill/MyGomocupStub
[10]: https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
[11]: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
