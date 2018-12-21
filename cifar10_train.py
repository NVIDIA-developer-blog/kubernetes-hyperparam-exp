from utils import *

#Network definition
def conv_bn(c_in, c_out, bn_weight_init=1.0, **kw):
    if kw['use_bn']:
        conv_block = {
            'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False), 
            'bn': batch_norm(c_out, bn_weight_init=bn_weight_init, **kw), 
            'relu': nn.ReLU(True),
                }
    else:
        conv_block = {
            'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False), 
            'relu': nn.ReLU(True),
                }

    return conv_block
    
def residual(c, **kw):
    return {
        'in': Identity(),
        'res1': conv_bn(c, c, **kw),
        'res2': conv_bn(c, c, **kw),
        'add': (Add(), [rel_path('in'), rel_path('res2', 'relu')]),
    }

def basic_net(channels, weight,  pool, **kw):
    return {
        'prep': conv_bn(3, channels['prep'], **kw),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1'], **kw), pool=pool),
        'layer2': dict(conv_bn(channels['layer1'], channels['layer2'], **kw), pool=pool),
        'layer3': dict(conv_bn(channels['layer2'], channels['layer3'], **kw), pool=pool),
        'classifier': {
            'pool': nn.MaxPool2d(4),
            'flatten': Flatten(),
            'linear': nn.Linear(channels['layer3'], 10, bias=False),
            'logits': Mul(weight),
        }
    }

def net(channels=None, weight=0.125, pool=nn.MaxPool2d(2), extra_layers=(), res_layers=('layer1', 'layer3'), **kw):
    channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
    n = basic_net(channels, weight, pool, **kw)
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], **kw)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer], **kw)       
    return n

def get_hyperparameters(job_id):
    # Update file name with correct path
    with open("hyperparams.yml", 'r') as stream:
        hyper_param_set = yaml.load(stream)
    print("\nHypermeter set for job_id: ",job_id)
    print("------------------------------------")
    pprint.pprint(hyper_param_set[job_id-1]["hyperparam_set"])
    print("------------------------------------\n")

    return hyper_param_set[job_id-1]["hyperparam_set"] 

losses = {
    'loss':  (nn.CrossEntropyLoss(reduction='sum'), [('classifier','logits'), ('target',)]),
    'correct': (Correct(), [('classifier','logits'), ('target',)]),
}

class TSVLogger():
    def __init__(self):
        self.log = ['epoch\tseconds\ttop1Accuracy']
    def append(self, output):
        epoch, hours, acc = output['epoch'], output['total time'], output['test acc']*100
        self.log.append(f'{epoch}\t{hours:.8f}\t{acc:.2f}')
    def __str__(self):
        return '\n'.join(self.log)
   
def main():
    job_id = int(os.environ['JOB_ID'])
    DATA_DIR = '/datasets/cifar10-data'

    #print('Downloading datasets')
    train_set_raw = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=False)
    test_set_raw = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=False)
    
    # Load hyperparameters
    hyperparams = get_hyperparameters(job_id)
    
    max_learning_rate = hyperparams["max_learning_rate"]
    data_aug_cutout_size = hyperparams["data_aug_cutout_size"]
    batch_size = hyperparams["batch_size"]
    momentum = hyperparams["momentum"]
    use_bn = hyperparams["batch_norm"]
    
    lr_schedule = PiecewiseLinear([0, 5, 24], [0, max_learning_rate, 0])
    
    model = TorchGraph(union(net(use_bn=use_bn), losses)).to(device).half()
    opt = nesterov(trainable_params(model), momentum=momentum, weight_decay=5e-4*batch_size)
        
    # print('Warming up cudnn on random inputs')
    for size in [batch_size, len(test_set_raw) % batch_size]:
        warmup_cudnn(model, size)
    
    t = Timer()
    train_set = list(zip(transpose(normalise(pad(train_set_raw.train_data, 4))), train_set_raw.train_labels))
    test_set = list(zip(transpose(normalise(test_set_raw.test_data)), test_set_raw.test_labels))

    TSV = TSVLogger()
    
    train_set_aug = Transform(train_set, [Crop(32, 32), FlipLR(), Cutout(data_aug_cutout_size, data_aug_cutout_size)])
    summary = train(model, lr_schedule, opt, train_set_aug, test_set, 
          batch_size=batch_size, loggers=(TableLogger(), TSV), timer=t, test_time_in_total=False, drop_last=True)
        
    with open('/datasets/results_job_id_'+str(job_id)+'.log', 'w') as csvfile:
        cw = csv.writer(csvfile, delimiter=',')
        for key, val in summary.items():
            cw.writerow([key, val])    
       
if __name__ == '__main__':
    main()

