import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler, SequentialSampler,WeightedRandomSampler
import multiprocessing
import os


class DNADataset(Dataset):

    def __init__(self, chromosomes, path=None):
        '''
        Accepts a list of chromosomes to load into its dataset. This is generally a list of 1 or more chromosome indices
        '''
        self.examples = []
        self.labels = []
        self.bins = []
        self.data, self.chrom_lengths = self.loadAll(chromosomes, path)


        for chrom in self.data:
            self.bins.extend(chrom[0])
            self.labels.extend(chrom[1])
            self.examples.extend(chrom[2])

    def __len__(self):
        return len(self.examples)

    #Returns the 4 channel example strings as a converted int lists, along with their label
    def __getitem__(self, idx):
        converted_example = self.convertChannel(self.examples[idx])
        return [self.bins[idx], self.labels[idx], converted_example]

    def convertChannel(self, example):
        channels = []
        for i in range (len(example)):
            channels.append( list( map( int, example[i])))

        return channels

    def get_label(self, idx):
        return self.labels[idx]

    def loadAll(self, chromosomes, path = None):
        substring = "rawPCAOneHot"
        dataset = []
        chromLengths = []
        binwiseData = []
        binwiseLabels = []
        binwiseBinID = []

        if path is not None:
            print("Loading from custom path:", path) 
            data_path = path
        else:
            data_path = './Data/'

        for chrom_index in chromosomes:
            print("Loading: " + str(chrom_index))
            with open(data_path + str(chrom_index+1) + substring + '.fa') as processedFile:
                details = processedFile.readline().split(',')

                # skip the first X unused bins
                #Read the rest into memory
                end_of_file = False
                num_examples = 0
                while end_of_file != True:

                    dataline = processedFile.readline()
                    if not dataline:
                        end_of_file = True
                    else: 
                        dataline = dataline.strip('\n').split(',')
                        if len(dataline) > 1:
                            num_examples += 1
                            # Set dataline[0] to be our expected output as a numerical float
                            binID = int(dataline[0])
                            label = float(dataline[1])

                            # converts the 4 channels of '010001' strings to lists of integers
                            channels = []
                            for channel in range(2, 6):
                                channels.append( dataline[channel])

                            # Append the resulting int converted training example to the dataset
                            binwiseBinID.append(binID)
                            binwiseLabels.append(label)
                            binwiseData.append(channels)

                print([num_examples, float(details[-1].strip('\n'))])

            dataset.append([binwiseBinID, binwiseLabels, binwiseData])
            chromLengths.append(len(binwiseLabels))
            binwiseData = []
            binwiseLabels = []
            binwiseBinID = []

        return dataset, chromLengths

class CustomSampler(Sampler[int]):

    #put variables here if they belong to the class

    def __init__(self, dataset, training_idx, generator=None) -> None:

        #put vars here if they belong to the object
        self.generator = generator
        self.sorted_training_A = []
        self.sorted_training_B = []
        self.current_A_idx = 0
        self.current_B_idx = 0
    
        for idx in training_idx:
            
            if float(dataset.labels[idx])> 0.5:
                self.sorted_training_A.append(idx)
            else:
                self.sorted_training_B.append(idx)

    #the arrow after self below is a function annotation
    def __iter__(self):

        final_training_idx = []

        numOfTrainIndices = min(len(self.sorted_training_A), len(self.sorted_training_B)) #get whichever is the smaller list of training indices

        loader_testing_outputList.append("The length of A: " + str(len(self.sorted_training_A))  + "  and the length of B: " + str(len(self.sorted_training_B)))
        for idx_num in range(numOfTrainIndices):

            """The purpose of the 2 if statements below is to make sure you use all data in the dataset, while accounting fot the fact that the 
            sorted_A and sorted_B are lists of different sizes, so instead of cutting one list short, we save where we were in the list and continue
            from there in the next epoch. Once all the data has been seen, then the index is reset to 0 and the sorted_list is shuffled
            """
            if self.current_A_idx >= len(self.sorted_training_A):
                loader_testing_outputList.append("********************A_RESET***************************")
                self.current_A_idx = 0
                np.random.shuffle(self.sorted_training_A)

            if self.current_B_idx >= len(self.sorted_training_B):
                loader_testing_outputList.append("********************B_RESET***************")
                self.current_B_idx = 0
                np.random.shuffle(self.sorted_training_B)
            
            loader_testing_outputList.append(self.sorted_training_A[self.current_A_idx])
            loader_testing_outputList.append(self.sorted_training_B[self.current_B_idx])

            final_training_idx.append(self.sorted_training_A[self.current_A_idx]) #add A to the final index list
            final_training_idx.append(self.sorted_training_B[self.current_B_idx]) #add B to the final index list

            self.current_A_idx += 1 #count up the indices for sorted_A
            self.current_B_idx += 1 #count up the indices for sorted_B
        
        self.balanced_indices = final_training_idx
        loader_testing_outputList.append('END')
        for i in range(len(self.balanced_indices)):
            yield self.balanced_indices[i]

    def __len__(self) -> int:
        return len(self.indices)

def custom_collate_fn(batch):
    bins = [[item[0]] for item in batch]
    labels = [[item[1]] for item in batch]
    examples = [item[2] for item in batch]
    return torch.Tensor(bins), torch.Tensor(labels), torch.Tensor(examples)

def getClassWeights(data,dataset_idx):

    #count the compartment consistency distribution
    #count_dict = {'A':0, 'B':0, 'Var':0}
    count_dict = {'A':0, 'B':0}

    
    for index in dataset_idx:
        """
        if data.labels[index] >= 0.8: #if the element is greater than 0.5, then its an A but if its less than 0.5, we can say its a B
            count_dict['A']+= 1
        elif data.labels[index] <= 0.2:
            count_dict['B']+= 0.5
        else:   
            count_dict['Var']+= 1
        """
        if data.labels[index] > 0.45: 
            count_dict['A']+= 1
        else:
            count_dict['B']+= 1

    counts = [i for i in count_dict.values()]
    counts_asnp = np.asarray(counts)
    class_weights = 1/counts_asnp 

    #now update the count_dict with weights instead of counts
    count_dict['A'] = class_weights[0]
    count_dict['B'] = class_weights[1]
    #count_dict['Var'] = class_weights[2]

    return count_dict

def assignWeights(dataset,dataset_idx,class_weights):

    sample_weights = [0] * len(dataset.labels)
    
    for idx in dataset_idx:
        """
        if dataset.labels[idx] >= 0.8:
            sample_weights[idx] = class_weights['A']      
        elif dataset.labels[idx] <= 0.2:
            sample_weights[idx] = class_weights['B']
        else:
            sample_weights[idx] = class_weights['Var']
        """
        if dataset.labels[idx] > 0.45:
            sample_weights[idx] = class_weights['A']      
        else:
            sample_weights[idx] = class_weights['B']

    return sample_weights

def setupBatches(dataset, training_idx, validation_idx, drop_last = True):

    sorted_training_A = []
    sorted_training_B = []
    sorted_validation_A = []
    sorted_validation_B = []

    for idx in training_idx:
        
        if float(dataset.labels[idx])> 0.5:
            sorted_training_A.append(idx)
        else:
            sorted_training_B.append(idx)

    
    for idx in validation_idx:

        if float(dataset.labels[idx]) > 0.5:
            sorted_validation_A.append(idx)
        else:
            sorted_validation_B.append(idx)
    

    #shuffle everything:
    np.random.shuffle(sorted_training_A)
    np.random.shuffle(sorted_training_B)
    np.random.shuffle(sorted_validation_A)
    np.random.shuffle(sorted_validation_B )

    final_training_idx = []
    final_validation_idx = []
    
    #if we don't want to drop the extra datapoints as the A and B compartment frequency is not exactly evenly distributed.
    if drop_last == False:

        numOfTrainIndices = max(len(sorted_training_A), len(sorted_training_B))
        numOfValIndices = max(len(sorted_validation_A), len(sorted_validation_B))
        idx_num = 0
        for idx_num in range(numOfTrainIndices):

            if idx_num < len(sorted_training_A):
                final_training_idx.append(sorted_training_A[idx_num])
            
            if idx_num < len(sorted_training_B):
                final_training_idx.append(sorted_training_B[idx_num])

        for idx_num in range(numOfValIndices):

            if idx_num < len(sorted_validation_A):
                final_validation_idx.append(sorted_validation_A[idx_num])
            
            if idx_num < len(sorted_validation_B):
                final_validation_idx.append(sorted_validation_B[idx_num])
    
    else: #drop the extra:

        numOfTrainIndices = min(len(sorted_training_A), len(sorted_training_B))
        numOfValIndices = min(len(sorted_validation_A), len(sorted_validation_B))
        idx_num = 0

        for idx_num in range(numOfTrainIndices):

            final_training_idx.append(sorted_training_A[idx_num])
            final_training_idx.append(sorted_training_B[idx_num])

        for idx_num in range(numOfValIndices):

            final_validation_idx.append(sorted_validation_A[idx_num])
            final_validation_idx.append(sorted_validation_B[idx_num])

    
    return final_training_idx, final_validation_idx

def getDataLoaders(withheld_test_chrom, args,file_path, chrom_limiter=22, shuffle_data = True, valid_size = 0.1):
    chroms = [i for i in range(chrom_limiter) if i != withheld_test_chrom]

    data = DNADataset(chroms, file_path + args['data_path'])
    data_test = DNADataset([withheld_test_chrom], file_path + args['data_path'])

    # train and valid
    num_trainval_data = len(data)
    trainval_idx = list(range(num_trainval_data))

    if shuffle_data:
        np.random.shuffle(trainval_idx)

    #basically here is where we seperate the training indices and the validation indicies. We seperate their indicies and by default we validate using 0.1 or 10% of the data
    split_tv = int(np.floor(valid_size * len(trainval_idx)))
    train_idx = trainval_idx[split_tv:]
    valid_idx = trainval_idx[:split_tv]

    #test
    num_test_data = len(data_test)
    test_idx = list(range(num_test_data))


    if shuffle_data:
        train_sampler = BatchSampler(CustomSampler(data,train_idx), args['batch_size'], args['drop_last'])
        valid_sampler = BatchSampler(SequentialSampler(valid_idx), args['batch_size'], args['drop_last'])

    else:
        train_sampler = BatchSampler(SequentialSampler(train_idx), args['batch_size'], args['drop_last'])
        valid_sampler = BatchSampler(SequentialSampler(valid_idx), args['batch_size'], args['drop_last'])

    test_sampler = BatchSampler(SequentialSampler(test_idx), args['batch_size'], args['drop_last'])

    print("dataloaders given " + str(args["data_loader_workers"]) + " workers..")
    train_loader = torch.utils.data.DataLoader(data, batch_sampler = train_sampler, collate_fn=custom_collate_fn, num_workers=args["data_loader_workers"])
    valid_loader = torch.utils.data.DataLoader(data, batch_sampler = valid_sampler, collate_fn=custom_collate_fn, num_workers=args["data_loader_workers"])
    test_loader = torch.utils.data.DataLoader(data_test, batch_sampler = test_sampler, collate_fn=custom_collate_fn, num_workers=args["data_loader_workers"])

    print("Dataloaders returned..")
    return train_loader, valid_loader, test_loader

if __name__ == '__main__':  
    print("testing data loader")  
    loader_testing_outputList = []

    args = {'batch_size':10, 'env':'mist', 'data_loader_workers':2, 'drop_last':True, 'data_path':'/Data/consistency/ProcessedOneHot_4c_gn_mESC/chr'}
    
    train_loader, valid_loader, test_loader = getDataLoaders(1,args,'.', chrom_limiter=1)


    count_A = 0
    count_B = 0
    test_list = []
    test_count = 0
    test_file = open(os.path.join(".\DNADataLoader_TestOutput_setupBatches.txt"), "w")
    
    print('TRAINING SIMULATION BEGINS: \n')
    for epoch in range(2):
        print("CURRENTLY IN EPOCH: " + str(epoch) + "\n")
        for data in train_loader:

            for value in data[0]:
                test_list.append(value)
                if value > 0.5:
                    count_A += 1
                else:
                    count_B += 1
            
            test_file.write('CLASS A:' + '\t' + str(count_A) + '\t' + 'CLASS B:' + '\t' + str(count_B) + '\n')
            count_A = 0
            count_B = 0
            test_count += 1
        
        test_file.write('BEGIN TEST LIST: \n')
        for value in loader_testing_outputList:
            test_file.write(str(value) + '\n')
        
        loader_testing_outputList.clear()

    #print(train_loader)
    #for label, example in train_loader:
        #print(label)
