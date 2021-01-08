
# TODO(yuchen): support CNN
# it only works for BERT now
WHITE_LIST = [
    'mul',
    'addv2',
    'batchmatmulv2',
    'square',
    'l2loss',
    'matmul',
    'sum',
    'tile',
    'sqrt',
    'transpose',
    'neg',
    'randomuniform',
    'cast',
    'greaterequal',
    'squareddifference',
    'softmax',
    'pow',
    'gatherv2',
    'onehot',
    'unsortedsegmentsum',
    'logsoftmax',
    'pad',
    'mean',
    'sub',
    'realdiv',
    'stridedslice',
]

# coefficient-wise operator 
# see https://eigen.tuxfamily.org/dox/group__TutorialArrayClass.html
CWISE_LIST = [
    'mul',
    'addv2',
    'square',
    'sqrt',
    'neg',
    'squareddifference',
    'pow',
    'sub',
    'realdiv'
]
