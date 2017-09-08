import os

cfg = type('', (), {})()

cfg.WORK_DIR = '/home/splendor/wd2t/fusor'

cfg.BRAIN_DIR = os.path.join(cfg.WORK_DIR, 'brain')
cfg.RL_BRAIN_DIR = os.path.join(cfg.WORK_DIR, 'rl_brain')
cfg.FILE_PREFIX = 'model.ckpt'
cfg.BRAIN_CHECKPOINT_FILE = os.path.join(cfg.BRAIN_DIR, cfg.FILE_PREFIX)
cfg.SUMMARY_DIR = os.path.join(cfg.WORK_DIR, 'summary')
cfg.REPLAY_MEMORY_DIR = os.path.join(cfg.WORK_DIR, 'replay')
cfg.STAT_FILE = os.path.join(cfg.WORK_DIR, 'stat.npz')
cfg.MID_VIS_FILE = os.path.join(cfg.WORK_DIR, 'mid_vis.npz')
cfg.DATA_SET_DIR = os.path.join(cfg.WORK_DIR, 'dataset_gomocup15')
cfg.DATA_SET_FILE = os.path.join(cfg.DATA_SET_DIR, 'train.txt')
cfg.DATA_SET_TRAIN = os.path.join(cfg.DATA_SET_DIR, 'train.txt')
cfg.DATA_SET_VALID = os.path.join(cfg.DATA_SET_DIR, 'validation.txt')
cfg.DATA_SET_TEST = os.path.join(cfg.DATA_SET_DIR, 'test.txt')

cfg.REPLAY_MEMORY_CAPACITY = 1000
cfg.REINFORCE_PERIOD = 500

cfg.TRAIN_EPOCHS = 10
cfg.FEED_BATCH_SIZE = 32
cfg.TRAIN_QUEUE_CAPACITY = 10000
cfg.VALIDATE_QUEUE_CAPACITY = 1000
cfg.SAMPLE_BATCH_NUM = 100
