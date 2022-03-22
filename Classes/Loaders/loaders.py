#from Classes.Dataset.BaseDataset import BaseDataset
from torch.utils.data import random_split, DataLoader
from torch import Generator

class BaseDataLoader():
    def __init__(self,  batch_size, dataset, seed, split = {'train': 0.7, 'val':0.15, 'test':0.15},):
        self.batch_size = batch_size
        self.split = split
        self.dataset = dataset

        train_size = int(split['train'] * len(dataset))
        test_size = int(split['test'] * len(dataset))
        val_size = len(dataset) - train_size - test_size

        '''
        train_dataset, test_val_dataset = random_split(dataset, [train_size, test_size + val_size])
        test_dataset, val_dataset = random_split(test_val_dataset, [test_size, val_size])
        '''

        train_dataset, test_dataset, val_dataset = random_split(
            dataset,
            [train_size, test_size, val_size],
            generator=Generator().manual_seed(seed))

        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validationloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        self.train_spit = split['train']
        self.train_loader = trainloader

        self.val_spit = split['val']
        self.test_loader = testloader

        self.test_spit = split['test']
        self.val_loader = validationloader




