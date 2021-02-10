
class DefaultConfig:
    def __init__(self):
        self.PROJECT_NAME = 'baseline'  # project name
        self.LOG_DIR = "./log"  # log directory
        self.OUTPUT_DIR = "./output"  # saved model directory
        self.DEVICE_ID = "0"  # GPU IDs, i.e. "0,1,2" for multiple GPUs

        self.LOG_PERIOD = 50  # iteration of displaying training log
        self.CHECKPOINT_PERIOD = 5  # saving model period
        self.EVAL_PERIOD = 1  # validation period
        self.MAX_EPOCHS = 30  # max training epochs

        # data
        self.DATALOADER_NUM_WORKERS = 4 # number of dataloader workers
        self.BATCH_SIZE = 12 #32#64  # MxN, M: number of persons, N: number of images of per person
        # model
        self.INPUT_SIZE = [512, 512]  # HxW
        self.MODEL_NAME = "resnet50"
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = "/xxx/pretrained_model/resnet50-19c8e357.pth"
        # loss
        self.LOSS_TYPE = 'BCE'
        self.LOSS_LABELSMOOTH = 'off'
        self.OPTIMIZER = 'Adam'  # optimizer
        self.BASE_LR = 0.00035  # base learning rate
        self.WEIGHT_DECAY = 0.0005
        self.BIAS_LR_FACTOR = 1.0
        self.WEIGHT_DECAY_BIAS = 0.0005
        self.MOMENTUM = 0.9
        self.STEPS = [15, 20, 25]
        self.GAMMA = 0.1  # decay factor of learning rate
        self.WARMUP_FACTOR = 0.01
        self.WARMUP_EPOCHS = 10  # warm up epochs
        self.WARMUP_METHOD = "linear"  # option: 'linear','constant'
        # test
        self.TEST_IMS_PER_BATCH = 32
        self.FEAT_NORM = "yes"
        self.TEST_WEIGHT = './output/resnet50_175.pth'
