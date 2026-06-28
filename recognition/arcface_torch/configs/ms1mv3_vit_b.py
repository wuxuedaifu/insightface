from easydict import EasyDict as edict

config = edict()
config.network = "vit_b_dp005_mask_005"
config.resume = False
config.output = None

config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.weight_decay = 0.1
config.batch_size = 128
config.lr = 1e-4
config.verbose = 2000
config.dali = False
config.optimizer = "adamw"

config.rec = "/train_tmp/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 40
config.warmup_epoch = 4
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
