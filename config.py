NUM_TRAIN = 50000
NUM_VAL   = 50000 - NUM_TRAIN
BATCH     = 128 # B
SUBSET    = 10000 # M
ADDENDUM  = 1000 # K

CYCLES = 10
INITIAL_LABELED = 1000
EPOCH = 200
LR = 0.1
GAMMA = 0.1

MILESTONES = [160]
EPOCHL = 120  # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model
MOMENTUM = 0.9
WDECAY = 5e-4
NUM_WORKERS = 4
RANDOM_SEED = 0