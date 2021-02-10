from .default import DefaultConfig

class Config(DefaultConfig):
    def __init__(self):
        super(Config, self).__init__()
        self.CFG_NAME = 'baseline'
        self.FINE_TUNE=None
        self.TEST_WEIGHT = ''
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.image_col_name = 'StudyInstanceUID'
        self.class_col_name = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                          'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
                          'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                          'Swan Ganz Catheter Present']
        self.paths = {'train_path': '../ranzcr-clip-catheter-line-classification/train',
                 'test_path': '../input/siim-isic-melanoma-classification/jpeg/test',
                 'csv_path': '../ranzcr-clip-catheter-line-classification/train.csv',
                 'log_path': './log.txt',
                 'save_path': './',
                 'model_weight_path_folder': '../input/efficientnet-weights'}