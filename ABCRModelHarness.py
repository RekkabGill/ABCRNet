import imp
from this import d
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from multiprocessing import Pool
import random
import sys
import os
import time
import DNADataLoader
from RMSE import RMSELoss
from DNADataLoader import DNADataset
from ABCNet import ABCNET


def printLogs(args, n_batches, file_path):
    if not os.path.exists(file_path + "/Results" + gridsearch_path):
        os.mkdir( file_path + "/Results" + gridsearch_path )
    f = open(os.path.join(file_path  + "/Results" + gridsearch_path, str(args["model_chromosome"]) + "," + str(args["instance_num"]) + "," + "TrainingLog.txt"), "a")
    f.write("\n")
    f.write("------- NEW RUN PARAMETERS -------\n")
    f.write("batch_size=" + str(args["batch_size"]) + '\n')
    f.write("epochs=" + str(args["epochs"]) + '\n')
    f.write("learning_rate=" + str(args["learning_rate"]) + '\n')
    f.close()

    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", args["batch_size"])
    print("epochs=", args["epochs"])
    print("learning_rate=", args["learning_rate"])
    print("NUMBER OF BATCHES PER EPOCH: ", n_batches)
    print("=" * 30)

def storeTrainingLoss(args, file_path, train_loss = -1.0, val_loss = -1.0):
    
    if train_loss != -1.0:
        with open(os.path.join(file_path + "/Results" + gridsearch_path, str(args["model_chromosome"]) + "," + str(args["instance_num"]) + "," + "TrainLoss.txt"), "a") as trainl:
            trainl.write(str(train_loss) + "," + str(val_loss) + "\n")

def getOptimizer(model, modelArgs):
    if modelArgs["optim"] == "SGD" :
        return torch.optim.SGD(model.parameters(), lr=modelArgs["learning_rate"], momentum=modelArgs["momentum"], nesterov=True)
    elif modelArgs["optim"] == "AdamW" :
        return torch.optim.AdamW(model.parameters(), lr=modelArgs["learning_rate"], betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    else:
        raise Exception("Unknown Optimizer specified")
    
def getLoss(modelArgs):
    if modelArgs['loss'] == 'MSE':
        return torch.nn.MSELoss()
    elif modelArgs['loss'] == 'L1':
        return torch.nn.L1Loss()
    elif modelArgs['loss'] == 'RMSE':
        return RMSELoss()
    else:
        raise Exception("Unknown Loss specified")

def runABCRTest(testType, model, dataLoader, epoch, args, file_path, return_preds=False):
    print("\nRunning " + testType + "...")

    model.eval() # Notify layers we are in test mode

    total_loss = [[]]
    predictions = []
    expectedOuts = []
    correspondingBins = []

    for binIDs, labels, inputs in dataLoader:
        # Move values to the GPU
        if args["use_cuda"]:
            binIDs = binIDs.cuda() 
            inputs = inputs.cuda()
            labels = labels.cuda() #THESE ARE THE GROUND TRUTH PCA VALUES (TRUTH CONSISTENCY PROBABILITIES)

        with torch.no_grad():
            classifications = model(inputs) #THESE ARE WHAT THE MODEL ACTUALLY OUTPUTS, WHAT IT THINKS THE CONSISTENCY PROBABILITIES ARE
            val_loss = model.loss(classifications, labels)

            total_loss[0].append(val_loss)

            # Gather statistics
            for i,classification in enumerate(classifications):
                if 'pred' in args['loss']:
                    classification = classification[0]

                if testType == "Testing":
                    correspondingBins.append(binIDs[i].item())
                    predictions.append(classification.item())
                    expectedOuts.append(labels[i].item())

    for c in range(len(total_loss)):
        total_loss[c] = sum(total_loss[c]) / len(total_loss[c])

    print(testType + " loss: ", total_loss)

    # Text file logging
    if not return_preds:
        f = open(os.path.join(file_path + "/Results" + gridsearch_path, str(args["model_chromosome"]) + "," + str(args["instance_num"]) + "," + "TrainingLog.txt"), "a")
        f.write("\n")
        f.write("****" + testType + " FOR EPOCH(train)/CHROMOSOME(test): " + str(epoch) + "\n")
        f.write(testType + " losses: " +  str(total_loss) + "\n")

        f.close()

        # Testing set never shuffled, so we record the predictions made for comparison to ground truth
        if testType == "Testing":
            with open(os.path.join(file_path + "/Results" + gridsearch_path, str(args["model_chromosome"]) + "," + str(args["instance_num"]) + "," + "TestPredictions.txt"), "w") as testf:

                for p in range(len(predictions)):
                    testf.write( str(correspondingBins[p]) + "," + str(predictions[p]) + "," + str(expectedOuts[p]) + "\n")

    if return_preds:
        return correspondingBins,predictions, expectedOuts
    return total_loss[0].item() #[tensor(0.1379, device='cuda:0')] <- output looks like this, so we use index 0 and the item() pulls out the decimal value

def trainABCRNet(model, dloader_train, dloader_val, dloader_test, args, file_path):
    #VARIABLE INITALIZATION:
    stoppingTrigger = 0 
    min_delta = 0.000
    best_validation = 100.111

    n_batches = len(dloader_train.dataset)/args["batch_size"]
    printLogs(args, n_batches, file_path)

    # Init the loss and optimizer functions
    model.loss = getLoss(args)
    model.optimizer = getOptimizer(model, args)

    print("Performing initial Validation Test")
    loss_val = runABCRTest("Validation", model, dloader_val, 0, args, file_path)

    # Loop over each epoch of training
    for epoch in range(args["epochs"]):
        print("Running Training...")
        model.train() # Notify layers we are in train mode

        running_predictions = []
        running_labels = []
        running_bins = []
        total_train_loss = [[]]
        print_every = int(1200/args["batch_size"]) #per training examples
        i = 0

        for binIDs, labels, inputs in dloader_train:
            i += 1
            # Move values to the GPU
            if args["use_cuda"]:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

            # Reset our gradients to zero
            model.optimizer.zero_grad()

            # Forward pass, backward pass and optimize
            outputs = model(inputs)
            running_predictions.extend(outputs.to('cpu'))
            running_labels.extend(labels.to('cpu'))
            running_bins.extend(binIDs)
            train_loss = model.loss(outputs,labels)
            train_loss.backward()
            model.optimizer.step()

            # Keep track of loss for statistics
            total_train_loss[0].append(train_loss.item())

            # print network status on occasion
            if (i + 1) % (print_every) == 0:
                num_batches = int(len(dloader_train.sampler) / args['batch_size'])
                print_train_loss = []
                for c in range(len(total_train_loss)):
                    print_train_loss.append([])
                    print_train_loss[c] = sum(total_train_loss[c]) / len(total_train_loss[c])
                print("Epoch {}, {:d}% \t train_loss: {:.4f}".format( epoch+1, int( 100 * (i+1) / num_batches ), print_train_loss[0]) )
                print("Last Output:", outputs[0][0], '\n', "Desired Output:", labels[0])

        training_loss = np.average(total_train_loss[0])
        print('The average training loss is: ', training_loss)

        #VALIDATION RUN AFTER EPOCH
        previous_loss = loss_val
        loss_val = runABCRTest("Validation", model, dloader_val, epoch+1, args, file_path)
        
        storeTrainingLoss(args, file_path, training_loss, loss_val)
        
        #SAVE THE BEST MODEL THAT YOU CAN TRAIN
        if(loss_val < best_validation):
            best_validation = loss_val
            print('new best validation loss value found: ' + str(best_validation) + ' saving model!')
            saveModel(model, args, "./Data/model_ckpts/ABCNet-" + args['model_name'] + "-Chr" + str(model_chromosome) + ".pt")

        if (args["use_early_stopping"]):

            #min_delta, stoppingTrigger = basicEarlyStoppingFunction(previous_loss, loss_val,stoppingTrigger, min_delta)
            stoppingTrigger = basicEarlyStoppingFunction(previous_loss, loss_val,stoppingTrigger)
            
            if stoppingTrigger >= args["stop_training_after"]:
                #activate early stopping
                print("EARLY STOPPING REACHED! Epoch: " + str(epoch))
                print("    Epochs since last validation accuracy reduction: " + str(stoppingTrigger))
                print("    Best Validation Loss achieved " + str(best_validation))
                print("    Curent Validation Loss " + str(loss_val))

                print('the value of stoppingTrigger is: ', stoppingTrigger)
                print('the value of args is: ',args["stop_training_after"])
                stoppingTrigger = 0
                min_delta = 0
                break #break out of the epoch

                      
    print("-"*30 + "\nTraining finished \nRunning Final Test...")

    # Run a test for withheld chromosome to evaluate its performance
    runABCRTest("Testing", model, dloader_test, i, args, file_path)
    return model

def earlyStoppingFunction(old_loss, current_loss, stoppingTrigger, min_delta):

    delta = old_loss - current_loss
    min_delta += delta 
    
    if ((args["learning_rate"] == 0.01 or args["learning_rate"] == 0.001) and min_delta >= 0.01):
        #reset
        print('resetting min delta and stoppingTrigger: ' + str(min_delta) + ' ' + str(stoppingTrigger) )
        return 0,0
    elif args["learning_rate"] == 0.0003 and min_delta >= 0.001:
        #reset
        print('resetting min delta and stoppingTrigger: ' + str(min_delta) + ' ' + str(stoppingTrigger) )
        return 0,0
    else:
        return min_delta, stoppingTrigger + 1

def basicEarlyStoppingFunction(old_loss, current_loss, stoppingTrigger):

    if current_loss > old_loss:
        return stoppingTrigger + 1
    else:
        return 0

def saveModel(model, args, PATH):
    '''
    Saves a completed network to disk
    '''
    torch.save([model.state_dict(), args], PATH)

def loadModel(model, PATH):
    '''
    Loads a network saved to disk
    '''
    ckpt = torch.load(PATH)
    model.load_state_dict(ckpt[0])
    return model, ckpt[1]

def loadArgs(args, arg_path):
    # DEFAULTS
    args["epochs"] = 24
    args["optim"] = "SGD"
    args["loss"] = 'MSE'
    args["learning_rate"] = 0.005
    args["momentum"] = 0.8
    args["use_early_stopping"] = False
    args["use_variable_LR"] = False
    args["batch_size"] = 32
    args["data_loader_workers"] = os.cpu_count()
    args['drop_last'] = True
    args["use_cuda"] = True
    args["reduce_LR_after"] = 0
    args["stop_training_after"] = 0

    argsfile = arg_path
    if os.path.exists(argsfile):
        with open(argsfile, 'r') as f:
            values = f.readlines()
            for line in values:
                value = line.split()
                if value[1].strip() == "True": # parse as true
                    args[value[0].strip()] = True
                elif value[1].strip() == "False": # parse as false
                    args[value[0].strip()] = False
                else:
                    try: # parse as int
                        args[value[0].strip()] = int(value[1].strip())
                    except:
                        try: # parse as float
                            args[value[0].strip()] = float(value[1].strip())
                        except: # parse as string
                            args[value[0].strip()] = value[1].strip()
    else:
        print("Args file not found.") 
    print(args)
    return args

#################################
########## MAIN  CODE ###########
#################################
if __name__ == '__main__':
    use_seed = True
    args = {}
    arg_file = "modelargs.txt"
    gridsearch_path = ""

    if len(sys.argv) > 2: # If we are using args, set the current test chrom and instance to them
        print("SLURM run detected")
        file_path = './Data/' + sys.argv[1]
        model_chromosome = int(float(sys.argv[2])) #The chromosome withheld for testing
        instance_num = model_chromosome #int(float(sys.argv[2])) The particular count of model being run for the given chromosome
        gridsearch_path = '/' + sys.argv[3]
        arg_file = "modelargs" + sys.argv[3] + ".txt"

    else:
        model_chromosome = 0 # We will use chr1 as our test set for single model training
        instance_num = 1 # Model instance number. Increment this if you want to train a second model with different logs tracked
        file_path = './Data/' + sys.argv[1]


    args["model_chromosome"] = model_chromosome
    args["instance_num"] = instance_num

    if torch.cuda.is_available():
        print("Cuda Available:", torch.cuda.get_device_name(0))
        cuda = torch.device('cuda')
        args["use_cuda"] = True
    else:
        print("Cuda UNAVAILABLE")
        args["use_cuda"] = False

    args = loadArgs(args, arg_file)

    #Make the randomization of the network static. All we are changing is which chromosome is withheld from the training/val set
    if (use_seed == True):
        manualSeed = 123
        np.random.seed(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print ("CHROMOSOME WITHHELD: ", str(model_chromosome))
    dloader_train, dloader_val, dloader_test = DNADataLoader.getDataLoaders(model_chromosome, args, file_path)

    ABCModel = ABCNET()
    if args["use_cuda"]:
        ABCModel = ABCModel.to('cuda')
    
    print("\nBeginning model training...")
    start = time.time()
    

    #create the model save location if it does not exist 
    if not os.path.exists("./Data/model_ckpts"):
        os.mkdir("./Data/model_ckpts")

    ABCModelTrained = trainABCRNet(ABCModel, dloader_train, dloader_val, dloader_test, args, file_path)
    #saveModel(ABCModelTrained, args, "./Data/model_ckpts/ABCNet-" + args['model_name'] + "-Chr" + str(model_chromosome) + ".pt")

    end = time.time()
    print("The elapsed time is: ", ((end - start)/3600), " hours")
    
    
