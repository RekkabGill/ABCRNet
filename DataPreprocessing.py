import random 
import numpy as np
import sys
import os
import multiprocessing as mp
import DNADataLoader

####
# An object containing a variety of functions for preprocessing data into parseable files
####
class PreprocessingCSV():
    globalCurrentBin = 0 #Keeps track of the range of bins belonging to each chromosome
    global_basepath = './Data/'
    compartment_file = 'compartment_file.txt'

    def __init__(self,base_path, file_name):
        self.global_basepath = './Data/' + base_path + '/'
        self.compartment_file = file_name

        if not os.path.exists(self.global_basepath):
            os.mkdir(self.global_basepath)

        if not os.path.exists(self.global_basepath + 'Processed/'):
            os.mkdir(self.global_basepath + 'Processed/')
        
    #Loads in the information and writes contents as "expected output, Custom character bin" per line
    def processCustomChunks(self, chrom, resolution_size, single_file):
        processedDataFile = open(self.global_basepath + 'Processed/' + chrom + 'rawPCA.fa', 'w')
        
        #First line of each processed chrom file looks like:
        # chr1,30,0,1954,0.486666
        # chr2,30,1955,3774,0.486666
        processedDataFile.write(chrom + ',' + str(self.getStartingBinValue_custom(chrom, resolution_size, single_file)) + ',' + str(self.globalCurrentBin) + ',' + str(self.globalCurrentBin + self.getEndingBinNumber_custom(chrom,resolution_size,single_file)) + '\n')

        self.globalCurrentBin += self.getEndingBinNumber_custom(chrom,resolution_size,single_file) +1

        count = 0 # Custom chunks of data equals 2000 lines
        chunkNum = 0 # To keep track of our progress
        chunkCustom = "" # Stores each bin of characters
        bin_size = (resolution_size / 50) - 1
            
        with open('./Data/Genome_Data/' + chrom + '.fa') as rawFile:
            next(rawFile) # Skip first line (labels)
            for line in rawFile:
                if (count < bin_size): # collect all the data into our bin
                    chunkCustom += line.strip()
                    count += 1
                    
                else:  # Add bin as line to our file
                    chunkCustom += line.strip()
                    count = 0
                    chunkNum += 1
                    expectedOut = self.getPCA_custom(chunkNum, chrom, resolution_size, single_file)
                    #print(chunkNum, expectedOut)
                    
                    if expectedOut != '0':
                      processedDataFile.write(str(chunkNum) + "," + expectedOut + "," + chunkCustom + '\n')

                    chunkCustom = ""
    
    # Gets the PCA value belonging to the current bin
    def getPCA_custom(self, chunkNum, chrom, resolution_size,single_file = False):

        if single_file == False:
            with open(self.global_basepath + 'Compartment_Data/mESC-' + chrom + '-pcaOut-res' + resolution_size + '.PC1.txt') as compFile:
                compFile.readline() # skip first line

                for line in compFile:
                    data = line.split()
                    
                    if (chunkNum*resolution_size >= int(data[1]) and chunkNum*resolution_size <= int(data[2])):
                        #check if the PCA is a nan value:
                        if data[3] == 'nan':
                            return '0'
                        else:
                            PCA = data[3]
                            return PCA
                return '0'

        else: #the compartment file is a single file
            with open(self.global_basepath + 'Compartment_Data/' + self.compartment_file) as compFile:
                compFile.readline() # skip first line

                for line in compFile:
                    data = line.split()
                    
                    if chrom == data[0]: #only process for the appropriate chromosome 
                        if (chunkNum*resolution_size >= int(data[1]) and chunkNum*resolution_size <= int(data[2])):
                            #check if the PCA is a nan value:
                            if data[3] == 'nan':
                                return '0'
                            else:
                                PCA = data[3]
                                return PCA
                return '0'
    
    #Gets the location of the first evaluated bin in the data (could range from bin 30 to bin 32)
    def getStartingBinValue_custom(self, chrom, resolution_size, single_file):

        if single_file == False:
            with open(self.global_basepath + 'Compartment_Data/mESC-' + chrom + '-pcaOut-res' + resolution_size + '.PC1.txt') as binFile:
                binFile.readline() # skip first line
                data = binFile.readline().split()

                firstbin = int( int(data[1])/resolution_size )
                return firstbin
        
        else: #its a single file 
            with open(self.global_basepath + 'Compartment_Data/' + self.compartment_file) as binFile:
                binFile.readline() # skip first line
                data = binFile.readline().split()

                firstbin = int( int(data[1])/resolution_size )
                return firstbin


    #Gets the total number of bins in the current dataset
    def getEndingBinNumber_custom(self, chrom, resolution_size, single_file):

        if single_file == False:
            with open(self.global_basepath + 'Compartment_Data/mESC-' + chrom + '-pcaOut-res' + resolution_size + '.PC1.txt') as binFile:
                elements = binFile.read().split()
                last_element = elements[len(elements)-3]
                last_bin = int( int(last_element)/resolution_size) 

                return last_bin

        else: #it is a single file

            with open(self.global_basepath + 'Compartment_Data/' + self.compartment_file) as binFile:

                for line in binFile: 
                    data = line.split()
                    chrom_label = data[0]

                    if chrom_label == '#bedGraph':
                        current_chrom = data[2].split(':')

                        if current_chrom[0] == chrom:
                            last_binVal = data[2].split('-')
                            last_bin = int (int(last_binVal[1]) / resolution_size)
                            return last_bin


    # Reprocesses all bins of data
    def reprocess(self,total_chrom, resolution_size, single_file):
        for label in range(total_chrom):
            print("Current Chrom: chr" + str(label+1))
            self.processCustomChunks("chr" + str(label+1), resolution_size, single_file)
        print("Reprocessing Completed...")

  
    # Helper function to convert chars to one hot
    def convertCharToOneHot(self, c):
        char = c.upper()
        
        if char == 'A': return ['0','0','0','1']
        elif char == 'T': return ['0','0','1','0']
        elif char == 'C': return ['0','1','0','0']
        elif char == 'G': return ['1','0','0','0']
        else: return ['0','0','0','0']


    def writeToOneHotFiles(self, chromSpecs):
        chromIdx = chromSpecs[0]
        folder_name = chromSpecs[2]
        print(self.global_basepath + folder_name + '/chr' + str(chromIdx) + "rawPCAOneHot.fa")

        with open(self.global_basepath + folder_name + '/chr' + str(chromIdx) + "rawPCAOneHot.fa", 'w') as newF:
            with open(self.global_basepath + 'Processed/chr' + str(chromIdx) + 'rawPCA.fa', 'r') as currF:
                details = currF.readline()
                print(details)
                newF.write(details)
                details = details.split(',')

                for lineIdx in range(int(details[1])):
                    currF.readline()

                for lineIdx in range(int(details[3]) - int(details[2]) - int(details[1])):
                    te = currF.readline().split(",")
                    if len(te) > 1:
                        newF.write(te[0] + ",") #this writes the bin ID
                        newF.write(te[1]) #this writes the expected output

                        channels = [[] for i in range(4)]

                        for char in te[2].strip():
                            cv = self.convertCharToOneHot(char)
                            for channel in range(4):
                                channels[channel].append(cv[channel])

                        for channel in range(4):
                            channels[channel] = ''.join(channels[channel])

                        write_str = ""
                        for channel in range(4):
                            write_str += "," + channels[channel]
                        write_str += '\n'
                        newF.write(write_str)

                        if lineIdx % 200 == 0:
                            print("Line: " + str(lineIdx) + " of Chrom: " + str(chromIdx))


    def mpOnehotEncoding(self, total_chrom, extra_names = ""):
        p = mp.Pool(6)
        
        # Init the folder outside of MP to avoid issues
        norm_type_code = 'g'
        folder_name = 'ProcessedOneHot_4c_' + norm_type_code + 'n'
        folder_name = folder_name + extra_names
        if not os.path.exists(os.path.join(self.global_basepath,folder_name)):
            os.mkdir(os.path.join(self.global_basepath,folder_name))

        chromSpecs = [[i, 4, folder_name] for i in range (1,(total_chrom+1))]

        print("Starting pooled conversions..")
        p.map(self.writeToOneHotFiles, chromSpecs)
        print("One Hot Encoding Completed...")


if __name__ == '__main__':

    base_path = sys.argv[1]
    file_name = sys.argv[2]

    p = PreprocessingCSV(base_path, file_name)
    extra_names = "_mESC" # An extra string to attach to the resulting file folders

    total_chrom = 22
    resolution_size = 250000
    single_file = True
    p.reprocess(total_chrom, resolution_size, single_file)
    p.mpOnehotEncoding(total_chrom, extra_names)
