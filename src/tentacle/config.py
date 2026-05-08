import os

cfg = type('', (), {})()

cfg.WORK_DIR = '.'

cfg.BRAIN_DIR = os.path.join(cfg.WORK_DIR, 'zero')
cfg.RL_BRAIN_DIR = os.path.join(cfg.WORK_DIR, 'rl_brain')
cfg.FILE_PREFIX = 'model.ckpt'
cfg.BRAIN_CHECKPOINT_FILE = os.path.join(cfg.BRAIN_DIR, cfg.FILE_PREFIX)
cfg.SUMMARY_DIR = os.path.join(cfg.WORK_DIR, 'summary')
cfg.REPLAY_MEMORY_DIR = os.path.join(cfg.WORK_DIR, 'replay')
cfg.BRAIN1_FILE = os.path.join(cfg.WORK_DIR, 'brain1.pt')
cfg.BRAIN2_FILE = os.path.join(cfg.WORK_DIR, 'brain2.pt')
cfg.GUI_MSG_BATCH = 64
cfg.MID_VIS_FILE = os.path.join(cfg.WORK_DIR, 'mid_vis.npz')
cfg.DATA_SET_DIR = os.path.join(cfg.WORK_DIR, 'data/alphagomoku/dataset_gomocup15')
cfg.DATA_SET_FILE = os.path.join(cfg.DATA_SET_DIR, 'train.txt')
cfg.DATA_SET_TRAIN = os.path.join(cfg.DATA_SET_DIR, 'train.txt')
cfg.DATA_SET_VALID = os.path.join(cfg.DATA_SET_DIR, 'validation.txt')
cfg.DATA_SET_TEST = os.path.join(cfg.DATA_SET_DIR, 'test.txt')

cfg.REPLAY_MEMORY_CAPACITY = 1000
cfg.REINFORCE_PERIOD = 100
# ``rl_train`` 单次前向在 GPU 上处理的最大局面数（跨多局拼接）；过大易 OOM。
# 配合梯度累加，单 chunk 的激活内存即峰值；32GB 卡上 1024 通常安全。
cfg.RL_POSITION_CHUNK = 1024

cfg.TRAIN_EPOCHS = 20
cfg.FEED_BATCH_SIZE = 64
cfg.TRAIN_QUEUE_CAPACITY = 10000
cfg.VALIDATE_QUEUE_CAPACITY = 1000
cfg.SAMPLE_BATCH_NUM = 100
cfg.DATA_SET_SUFFIX = '_a_proportional.tfrecords'
