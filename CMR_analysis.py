'''
Various Functions
Date: Jan 2025
'''
import pandas as pd
import numpy as np
import sys 
import csv
import os
import math
import copy
import statistics
import scipy
from matplotlib import pyplot as plt 
from statistics import mean
from scipy import stats
from sklearn import metrics
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter


def everyBin_Analysis():

    file_path1 = "./data_files/human/human_compartment_files/" #HUMAN
    file_list = os.listdir(file_path1)

    #HUMAN:
    headers_list = ['#bedGraph section chr1:0-248956422', '#bedGraph section chr2:0-242193529', '#bedGraph section chr3:0-198295559','#bedGraph section chr4:0-190214555','#bedGraph section chr5:0-181538259', '#bedGraph section chr6:0-170805979', 
                        '#bedGraph section chr7:0-159345973', '#bedGraph section chr8:0-145138636', '#bedGraph section chr9:0-138394717', '#bedGraph section chr10:0-133797422', '#bedGraph section chr11:0-135086622', '#bedGraph section chr12:0-133275309',
                        '#bedGraph section chr13:0-114364328', '#bedGraph section chr14:0-107043718', '#bedGraph section chr15:0-101991189', '#bedGraph section chr16:0-90338345', '#bedGraph section chr17:0-83257441', '#bedGraph section chr18:0-80373285', 
                        '#bedGraph section chr19:0-58617616', '#bedGraph section chr20:0-64444167', '#bedGraph section chr21:0-46709983', '#bedGraph section chr22:0-50818468']

    #next let's get the starting bin values for this chrom:
    starting_df = pd.read_csv(file_path1 + file_list[0], sep = "\t", header = None, comment = '#')
    starting_df.columns = ['chromosome', 'start', 'end', 'datavalue']

    for chrom in range (1,23):

        master_list = []

        if chrom > 9: 
            chrom_string = str(chrom)
        else:
            chrom_string = '0' + str(chrom)
        
        print ('working on: chr' + chrom_string)

        bin_analysis_file = open('./output/human/compartments_per_bin/chr' + chrom_string + '_CompsPerBin.csv', 'w', newline="")
        writer = csv.writer(bin_analysis_file)
    

        starting_df_chr = starting_df[starting_df['chromosome'] == ('chr'+ str(chrom))].copy()
        starting_bins = list(starting_df_chr['start'])
        starting_bins.insert(0,'')

        #write out the bin numbers in the csv file
        writer.writerow(starting_bins)
        
        #Next handle the compartment values
        for idx, compartment_file in enumerate(file_list):
            aORb_list = []
            aORb_list.append(compartment_file)
            the_file = file_path1 + compartment_file
            dataframe1 = pd.read_csv(the_file, sep = "\t", header = None, comment = '#')
            dataframe1.columns = ['chromosome', 'start', 'end', 'datavalue']
        

            dataframe1_chr = dataframe1[dataframe1['chromosome'] == ('chr'+ str(chrom))].copy() #the copy is necessary to prevent the settingwithcopy warning (it flags confusing chained assignments. i.e. are you referencing original or new dataframe)
            compartment_values = list(dataframe1_chr['datavalue'])
            
            for compartment_val in compartment_values:
                
                if compartment_val > 0:
                    aORb_list.append('A')
                   
                elif compartment_val < 0:
                    aORb_list.append('B')
                
                else:
                    aORb_list.append('NaN')
            

            master_list.append(aORb_list)

        change_col_to_row = zip(master_list)
        
        for ls in change_col_to_row:

            for row in ls:
                writer.writerow(row) 

def everyBin_AnalysisRAW():

    file_path1 = "./data/human/human_compartment_files/" #HUMAN
    file_list = os.listdir(file_path1)

    #HUMAN:
    headers_list = ['#bedGraph section chr1:0-248956422', '#bedGraph section chr2:0-242193529', '#bedGraph section chr3:0-198295559','#bedGraph section chr4:0-190214555','#bedGraph section chr5:0-181538259', '#bedGraph section chr6:0-170805979', 
                        '#bedGraph section chr7:0-159345973', '#bedGraph section chr8:0-145138636', '#bedGraph section chr9:0-138394717', '#bedGraph section chr10:0-133797422', '#bedGraph section chr11:0-135086622', '#bedGraph section chr12:0-133275309',
                        '#bedGraph section chr13:0-114364328', '#bedGraph section chr14:0-107043718', '#bedGraph section chr15:0-101991189', '#bedGraph section chr16:0-90338345', '#bedGraph section chr17:0-83257441', '#bedGraph section chr18:0-80373285', 
                        '#bedGraph section chr19:0-58617616', '#bedGraph section chr20:0-64444167', '#bedGraph section chr21:0-46709983', '#bedGraph section chr22:0-50818468']

    #next let's get the starting bin values for this chrom:
    starting_df = pd.read_csv(file_path1 + file_list[0], sep = "\t", header = None, comment = '#')
    starting_df.columns = ['chromosome', 'start', 'end', 'datavalue']

    for chrom in range (1,23):

        master_list = []

        if chrom > 9: 
            chrom_string = str(chrom)
        else:
            chrom_string = '0' + str(chrom)
        
        print ('working on: chr' + chrom_string)

        bin_analysis_file = open('./output/human/compartments_per_binRAW/chr' + chrom_string + '_CompsPerBinRAW.csv', 'w', newline="")
        writer = csv.writer(bin_analysis_file)
    

        starting_df_chr = starting_df[starting_df['chromosome'] == ('chr'+ str(chrom))].copy()
        starting_bins = list(starting_df_chr['start'])
        starting_bins.insert(0,'')

        #write out the bin numbers in the csv file
        writer.writerow(starting_bins)
        
        #Next handle the compartment values
        for idx, compartment_file in enumerate(file_list):
            aORb_list = []
            aORb_list.append(compartment_file)
            the_file = file_path1 + compartment_file
            dataframe1 = pd.read_csv(the_file, sep = "\t", header = None, comment = '#')
            dataframe1.columns = ['chromosome', 'start', 'end', 'datavalue']
        

            dataframe1_chr = dataframe1[dataframe1['chromosome'] == ('chr'+ str(chrom))].copy() #the copy is necessary to prevent the settingwithcopy warning (it flags confusing chained assignments. i.e. are you referencing original or new dataframe)
            compartment_values = list(dataframe1_chr['datavalue'])
            
            for compartment_val in compartment_values:
                
                aORb_list.append(compartment_val)
            
            master_list.append(aORb_list)

        change_col_to_row = zip(master_list)
        
        for ls in change_col_to_row:

            for row in ls:
                writer.writerow(row)                

def compareGC_w_Error():

    gc_content_df = pd.read_csv(GC_CONTENT_PATH, sep = ",", header=None, comment= '#')
    gc_content_df.columns = ['chromosome', 'start', 'end','A_CMR', 'B_CMR','GC', 'AT']
    CUSTOM_CHROMOSOMES = [0]

    for chromosome in CUSTOM_CHROMOSOMES:

        if (chromosome+1) < 10:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + '0' + str((chromosome + 1)) + 'TestPredictions.txt'
        else:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + str((chromosome + 1)) + 'TestPredictions.txt'

        chrom = 'chr' + str(chromosome + 1) #we cant use the getChrom function because the file does match the '01' format, isnteads its just '1'
        print('working on: ', chrom)
        predictions_df = pd.read_csv(chromosome_path, sep = ",", header=None, comment= '#')
        predictions_df.columns = ['bin','predictions', 'targets']
        thePredictions = predictions_df['predictions'].to_list()
        #thePredictions = predictions_df['predictions'].to_numpy()

        gc_df_chr = gc_content_df[gc_content_df['chromosome'] == chrom].copy()
        gc_df_chr.dropna(inplace=True) #remove any of the nan from the gc content file (in the current chromosome)

        A_CMRs_fromPred = [round(val, 4) for val in predictions_df['targets'].to_list()] #round to 4 decimald digits
        A_CMRs_fromGC = [round(val,4) for val in gc_df_chr['A_CMR'].to_list()]
        gc_content = gc_df_chr['GC'].to_list()


        #LETS CALCLATE OUR GC CONTENT AND AVERAGE ERROR IN OUR COLUMNS of 0.1 - 0.2 - 0.3 - ... - 0.9
        prob1_2 = []
        prob8_9 = []

        for idx, target in enumerate(A_CMRs_fromPred):

            avg_error = 0
            #if chrom == 'chr2':
                #print(gc_content_df)
                #print(target)
                #print(A_CMRs_fromGC)
            if target >= 0.1 and target <=0.2:
                #print(target, idx)
                #print(A_CMRs_fromGC[idx])
                if target == A_CMRs_fromGC[idx]:
                    avg_error = abs(target - thePredictions[idx])
                    prob1_2.append([avg_error,gc_content[idx]])
                else:
                    raise Exception('The target value and the  value do not match in GC content Function')
                
            elif target >= 0.8 and target <= 0.9:
                if target == A_CMRs_fromGC[idx]:
                    avg_error = abs(target - thePredictions[idx])
                    prob8_9.append([avg_error,gc_content[idx]])
                else:
                    raise Exception('The target value and the gc value do not match in GC content Function')

        prob1_2errors,prob1_2gc = map(list,zip(*prob1_2))

        '''
        print('###################################################################################')
        print('Here are the statistics of Targets between 0.1 and 0.2 and their relative GC error:')
        print('min: ',min(prob1_2errors))
        print('max: ',max(prob1_2errors))
        print('Average: ',mean(prob1_2errors))
        print('Median: ',statistics.median(prob1_2errors))
        print('\n')

        prob8_9errors,prob8_9gc = map(list,zip(*prob8_9))
        print('###################################################################################')
        print('Here are the statistics of Targets between 0.8 and 0.9 and their relative GC error:')
        print('min: ',min(prob8_9errors))
        print('max: ',max(prob8_9errors))
        print('Average: ',mean(prob8_9errors))
        print('Median:', statistics.median(prob8_9errors))
        print('\n')
        '''

        #adjust the error_range below to your preference, some ranges won't work depending on the error distribution. The error_columns, set the x-axis labels for the bar graph
        error_range12 = [0.01,0.1]
        #error_range89 = [0.01,0.1,0.25,0.7]
        error_range89 = [0.01,0.1]
        error_columns12 = ['<0.01','<0.1', '>0.1']
        #error_columns89 = ['<0.01', '<0.1', '0.25', '<0.7']
        error_columns89 = ['<0.01','<0.1', '>0.1']

        er_gc_tiny = []
        er_gc_small = []
        er_gc_medium = []
        er_gc_large = []
        er_gc_big = []
        er_gc_huge = []
        median_list = []

        for unit in prob1_2:
            
            abs_error = unit[0]
            median_list.append(abs_error)

            #if abs_error < error_range12[0]:
            #    er_gc_tiny.append(unit[1])
            #else: 
            #    er_gc_small.append(unit[1])

            if abs_error < error_range12[0]:
                er_gc_tiny.append(unit[1])

            elif abs_error < error_range12[1]:
                er_gc_small.append(unit[1])

            elif abs_error >= error_range12[1]:
                er_gc_medium.append(unit[1])

            #elif abs_error >= error_range12[2]:
            #    er_gc_large.append(unit[1])
            
            #elif abs_error < error_range12[4]:
            #    er_gc_big.append(unit[1])
            
            #elif abs_error < error_range12[5]:
            #    er_gc_huge.append(unit[1])
            else: 
                raise Exception("There is an error with your error range at 0.1 to 0.2, check to make sure it can be binned properly.")

        avg_gc_er = []
        
        #the average gc content for that particular block of error i.e. < 0.01
        print('the 0.1-0.2 Median: ',statistics.median(median_list))
        avg_gc_er.append(mean(er_gc_tiny)) 
        avg_gc_er.append(mean(er_gc_small))
        avg_gc_er.append(mean(er_gc_medium))
        #avg_gc_er.append(mean(er_gc_large))
        #avg_gc_er.append(mean(er_gc_big))
        #avg_gc_er.append(mean(er_gc_huge))

        generateGCErrorPlot(avg_gc_er, chrom, '0.1 - 0.2', error_columns12,error_range12)

        ##################################NEXT DO THE 0.8 to 0.9 BIN AFTER CLEARING LISTS:
        er_gc_tiny.clear()
        er_gc_small.clear()
        er_gc_medium.clear()
        er_gc_large.clear()
        er_gc_big.clear()
        er_gc_huge.clear()
        avg_gc_er.clear()
        median_list.clear()

        for unit in prob8_9:
            
            abs_error = unit[0]
            median_list.append(abs_error)

            #if abs_error < error_range89[0]:
            #    er_gc_tiny.append(unit[1])
            #else: 
            #    er_gc_small.append(unit[1])

            if abs_error < error_range89[0]:
                er_gc_tiny.append(unit[1])

            elif abs_error < error_range89[1]:
                er_gc_small.append(unit[1])

            elif abs_error >= error_range89[1]:
                er_gc_medium.append(unit[1])

            #elif abs_error >= error_range89[2]:
            #    er_gc_large.append(unit[1])
            
            #elif abs_error < error_range89[4]:
            #    er_gc_big.append(unit[1])
            
            #elif abs_error < error_range89[5]:
            #    er_gc_huge.append(unit[1])
            
            else:
                raise Exception("There is an error with your error range at 0.8 to 0.9, check to make sure it can be binned properly.")

        print('the 0.8-0.9 Median: ',statistics.median(median_list))
        avg_gc_er.append(mean(er_gc_tiny))
        avg_gc_er.append(mean(er_gc_small))
        avg_gc_er.append(mean(er_gc_medium))
        #avg_gc_er.append(mean(er_gc_large))
        #avg_gc_er.append(mean(er_gc_big))
        #avg_gc_er.append(mean(er_gc_huge))

        generateGCErrorPlot(avg_gc_er, chrom, '0.8 - 0.9',error_columns89,error_range89)

        er_gc_tiny.clear()
        er_gc_small.clear()
        er_gc_medium.clear()
        er_gc_large.clear()
        er_gc_big.clear()
        er_gc_huge.clear()
        avg_gc_er.clear()
        median_list.clear()

def generateGCErrorPlot(gc_error_list, chrom, freq_section,error_columns,error_range):

    print("     CMR Target Range: ",freq_section)
    for idx,error_segement in enumerate(gc_error_list):
        print("     Error: ",error_columns[idx],", GC= ",error_segement,"|")

    #Overall Chart
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='col')
    fig.set_figheight(11.25) #1080 pixels
    fig.set_figwidth(20) #1920 pixels
    #fig.tight_layout(pad = 2)
    #plt.subplots_adjust(wspace = 0.1, hspace = 0.2)


    ax.set_xlabel('CMR Error Range', fontsize = 26,labelpad= 20)
    ax.set_ylabel('ABCRNet Input GC-content', fontsize = 26,labelpad= 20)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    ax.tick_params(axis = 'x', labelsize = 24, length = 5, pad = 10)
    ax.tick_params(axis = 'y', labelsize = 24)
    #title_string = "GC-Content vs. CMR Error for CMR Target Range (" + freq_section +") " + ANIMAL + " " + chrom
    #ax.set_title(title_string, y= 1, loc = 'center', fontsize = 28)
    ax.set_ylim([0,0.65])
    ax.set_yticks(np.arange(0,0.65,0.05))
    ax.bar(error_columns, gc_error_list, color = ['red'] , label = 'Average GC value', clip_on = False) # axis_ticks used here to create standard diagonal line
    ax.grid(axis = 'y', linewidth = 0.2)
    ax.margins(0)
    plt.savefig( (GCPLOT_SAVE_PATH + chrom +"_GCvsER_" + freq_section + "_" + ANIMAL + ".png" ), dpi = 300, bbox_inches='tight', pad_inches=0)

def getChromAsString(chromosome):

    if chromosome < 10:
        return 'chr0' + str(chromosome)
    else:
        return 'chr' + str(chromosome)
         
def predictionsBoxPlot(chromosome):

    df1 = pd.read_csv(SINGLE_CHR_PRED_PATH, sep = ",", header=None, comment= '#')
    df1.columns = ['bin','predictions', 'targets']

    thePredictions = df1['predictions'].to_numpy()
    theTargets = df1['targets'].to_numpy()

    chrom = getChromAsString(chromosome)
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='col')
    #fig.tight_layout(pad = 2)
    fig.set_figheight(11.25) #1080 pixels
    fig.set_figwidth(20) #1920 pixels
    #plt.subplots_adjust(wspace = 0.2, hspace = 0.4)

    ax.set_title("Boxplots of the predictions for "+ ANIMAL + " " + chrom , y= 1, loc = 'center', fontsize = 28)
    ax.set_ylabel('Relative Frequency (out of 55 Datasets)', fontsize = 30,labelpad= 12.0)
    ax.boxplot([theTargets, thePredictions], labels =['Targets', 'Predictions'])
    ax.tick_params(axis = 'both', labelsize = 24)
    ax.set_yticks(np.arange(0,1.1, 0.1))
    plt.grid(axis = 'y', linewidth = 0.2)
    #plt.show()
    plt.savefig( (BOXPLOTS_SAVE_PATH + ANIMAL +"_PredvsTarget_"+ chrom + ".png" ), dpi = 300)

def manyBoxPlots():

    for chromosome in range(CHROMOSOMES):

        if (chromosome+1) < 10:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + '0' + str((chromosome + 1)) + 'TestPredictions.txt'
        else:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + str((chromosome + 1)) + 'TestPredictions.txt'

        chrom = 'chr' + str(chromosome + 1) #we cant use the getChrom function because the file does match the '01' format, isnteads its just '1'
        print('working on: ', chrom)

        df1 = pd.read_csv(chromosome_path, sep = ",", header=None, comment= '#')
        df1.columns = ['bin','predictions', 'targets']

        thePredictions = df1['predictions'].to_list()
        theTargets = df1['targets'].to_list()
        target_dict = {}

        #first setup the dictionary keys for the entire range of target CMR's, so that later the entire range is represented in the graph, 
        #regardless if it had a collection of predictions or not (some of these target CMR's will not have predictions, i.e., a ratio that never occurs)
        '''
        cmr_range = np.arange(0.1,0.92,0.02)
        for x in cmr_range:
            rounded_x = round(x,2)
            target_dict[rounded_x] = []
        '''
        for idx, key in enumerate(theTargets):
            rounded_key = round(key,2)
            if rounded_key not in target_dict:

                target_dict[rounded_key] = []

            target_dict[rounded_key].append(thePredictions[idx])
        
        
        sorted_dict = dict(sorted(target_dict.items()))
        key_list = []
        value_list = []

        #key = targets, value = predicted values
        for key,value in sorted_dict.items():

            key_list.append(key)
            value_list.append(value)

        fig, ax = plt.subplots(nrows=1, ncols=1, sharex='col')
        #plt.gca().set_aspect(0.08)

        #fig.tight_layout(pad = 3)
        fig.set_figheight(10) #1080 pixels
        fig.set_figwidth(8) #1920 pixels
        #plt.subplots_adjust(wspace = 0.2, hspace = 0.4)
        #ax.set_aspect(1)

        #title_string = "Distributed predicted CMR vs. target CMR for " + ANIMAL + " " + chrom
        #ax.set_title(title_string, y= 1, loc = 'center', fontsize = 20)
        ax.set_ylabel('Target CMR', fontsize = 18,labelpad= 25,weight='bold')
        ax.set_xlabel('Distributed Predicted CMR', fontsize = 18,labelpad= 25, weight = 'bold')
        ax.boxplot(value_list,labels = key_list,vert=False)
        #ax.set_yticklabels(key_list)


        #turn all tick labels off first
        labels = ax.yaxis.get_ticklabels()
        for label in labels:
            label.set_visible(False)
        
        #only show nth tick label, gets too crowded the 
        #the [::3] means 'nothing for the first argument, nothing for the second, and jump by three'. It gets every third item of the sequence sliced!
        for label in labels[::3]:
            label.set_visible(True)
        
        for tickmark in ax.yaxis.get_ticklines():
            tickmark.set_visible(False)
        
        for tickmark in ax.yaxis.get_ticklines()[::3]:
            tickmark.set_visible(True)
        '''
        '''
        ax.tick_params(axis = 'both', labelsize = 14, pad=10, top=False,right=False)    
        ax.set_xlim([0,1])
        #ax.set_xticks(np.arange(0.1,1.1,0.1))
        #ax.set_yticks(np.arange(0,1.1, 0.1))
        plt.grid(linewidth = 0.2)
        #plt.show()
        plt.savefig( (BOXPLOTS_SAVE_PATH + chrom +"_manyboxplots_" + ".png" ), dpi = 300, bbox_inches='tight', pad_inches=0)

def predictionsScatterPlot(chromosome):

    chrom = getChromAsString(chromosome)

    dataframe1 = pd.read_csv(SINGLE_CHR_PRED_PATH, sep = ",", header = None, comment = '#')
    dataframe1.columns = ['predictions', 'groundtruth']

    thePredictions = dataframe1['predictions'].to_numpy()
    theTargets = dataframe1['groundtruth'].to_numpy()

    rmse_loss = 0.1579

    #Overall Chart
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='col')
    fig.tight_layout(pad = 2)
    plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    axis_ticks = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]


    ax.set_xlabel('Observed', fontsize = 30,labelpad= 25)
    ax.set_ylabel('Predictions', fontsize = 30,labelpad= 12.0)
    #ax[0].spines['top'].set_visible(False)
    #ax[0].spines['right'].set_visible(False)
    #ax[0].spines['bottom'].set_visible(False)
    ax.tick_params(axis = 'x', labelsize = 24, length = 0, pad=15)
    ax.tick_params(axis = 'y', labelsize = 24)
    #ax.set_title("Model Trained on N's Predicting chromosome without Ns -- "+ ANIMAL + chrom, y= 1, loc = 'center', fontsize = 32)
    ax.set_title("Model Trained on N's Predicting chromosome with Ns -- "+ ANIMAL + chrom, y= 1, loc = 'center', fontsize = 32)
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    ax.set_yticks(np.arange(min(axis_ticks), max(axis_ticks), 0.1))
    ax.set_xticks(np.arange(min(axis_ticks), max(axis_ticks), 0.1))
    ax.plot(axis_ticks, axis_ticks, color = 'r', label = 'Perfect Predictions') # axis_ticks used here to create standard diagonal line
    l1 = ax.scatter(theTargets, thePredictions, s =15, c= 'b', label = 'Predictions Vs Observed')

    stats_box = '\n'.join((r'$RMSE=%.3f$' % (rmse_loss, ),))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.80, 0.05, stats_box, transform=ax.transAxes, fontsize=28, verticalalignment='bottom',horizontalalignment='left', bbox=props)
    #ax[0].axhline(y=0.5, color = 'black', linewidth = 0.5)
    ax.legend(prop={'size':14})
    ax.margins(0)
    plt.show()

def getPredictionsFileName(chromosome):
    
    chrom = chromosome + 1
    
    if chrom < 10:
        file_name = '0' + str(chrom) + 'TestPredictions.txt'
    else:
        file_name = str(chrom) + 'TestPredictions.txt'
    
    return file_name

def ManyPredictionScatterPlots(theAnimal):
    if theAnimal== 'human': chromosomes = 22;
    else: chromosomes = 19 

    for chromosome in range(chromosomes): #HUMAN=22, MOUSE=19

        print('working on chromosome: ' + str(chromosome + 1))

        chrom = getChromAsString(chromosome+1)
        filePath = SINGLE_CHR_PRED_PATH
        dataframe1 = pd.read_csv(filePath, sep = ",", header = None, comment = '#')
        dataframe1.columns = ['bin','predictions', 'groundtruth']

        thePredictions = dataframe1['predictions'].to_numpy()
        theTargets = dataframe1['groundtruth'].to_numpy()


        #Overall Chart
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex='col')
        fig.tight_layout(pad = 2)
        fig.set_figheight(11.25) #1080 pixels
        fig.set_figwidth(20) #1920 pixels
        plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
        axis_ticks = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]


        ax.set_xlabel('Targets', fontsize = 30,labelpad= 25)
        ax.set_ylabel('Predictions', fontsize = 30,labelpad= 12.0)
        #ax[0].spines['top'].set_visible(False)
        #ax[0].spines['right'].set_visible(False)
        #ax[0].spines['bottom'].set_visible(False)
        ax.tick_params(axis = 'x', labelsize = 24, length = 0)
        ax.tick_params(axis = 'y', labelsize = 24)
        ax.set_title("No_N_Model: "+ theAnimal +'_'+ chrom, y= 1, loc = 'center', fontsize = 32)
        ax.set_ylim([0,1])
        ax.set_xlim([0,1])
        ax.set_yticks(np.arange(min(axis_ticks), max(axis_ticks), 0.1))
        ax.set_xticks(np.arange(min(axis_ticks), max(axis_ticks), 0.1))
        ax.plot(axis_ticks, axis_ticks, color = 'r', label = 'perfect_predictions') # axis_ticks used here to create standard diagonal line
        l1 = ax.scatter(theTargets, thePredictions, s =15, c= 'b', label = 'predictions_vs_targets')
        #ax[0].axhline(y=0.5, color = 'black', linewidth = 0.5)
        ax.margins(0)
        plt.savefig( (SCATTERPLOTS_SAVE_PATH + theAnimal +"_"+ 'No_N_Model_' + chrom + ".png" ), dpi = 300)
        plt.clf() #close the current figure that's open
        #plt.show()
    
    plt.close('all') #closes all current open figures to not overuse memory

def renameTrainingOutputFiles():

    file_list = os.listdir(PATH_TO_RENAMING)

    for file in file_list:
        print(file)

        file_name_list = file.split(',')
        chromosome = int(file_name_list[1]) + 1
        newFileName10 = str(chromosome) + file_name_list[2]
        newFileName = '0' + str(chromosome) + file_name_list[2] 

        if newFileName not in file_list and newFileName10 not in file_list:
            if chromosome < 10:    
                os.rename(PATH_TO_RENAMING + file, PATH_TO_RENAMING + '0' + str(chromosome) + file_name_list[2] )
            else:
                os.rename(PATH_TO_RENAMING + file, PATH_TO_RENAMING + str(chromosome) + file_name_list[2] )

def examineChromFiles():

    for chrom in range(1,23):
        
        end_of_file = False
        PATH_TO_CHROM = './chr' + str(chrom) + 'rawPCAOneHot.fa'
        print("Loading: " + str(chrom))

        with open(PATH_TO_CHROM) as processedFile:

            details = processedFile.readline()
            
            while end_of_file != True:
                
                data_line = processedFile.readline().split(',')
                print(len(data_line[1]))
                print(len(data_line[2]))
                print(len(data_line[3]))
                print(len(data_line[4]))

                if data_line == False:
                    end_of_file = True
                 
                print('the length of the one hot encoded string is: ' + str(len(data_line[1]) + len(data_line[2]) + len(data_line[3]) + len(data_line[4])))
                print('it should be: 1 000 000')
                quit(0)

def nModelPerformanceComparison(chromosome,the_title):

    chrom = getChromAsString(chromosome)

    nModel_df = pd.read_csv(COMBINENMODEL_PATH, sep = ",", header = None, comment = '#')
    nModel_df.columns = ['nBin','targets','pred_noNModel', 'pred_nModel']
    nModel_df = nModel_df.fillna(0)

  
    all_noN_bins = nModel_df[nModel_df['nBin'] == 0].copy()
    nModel_preds = all_noN_bins['pred_nModel'].to_numpy()
    noNModel_preds = all_noN_bins['pred_noNModel'].to_numpy()

    #test = np.sort(nModel_preds)
    #print(test)
    
    for idx,x in enumerate(nModel_preds):

        if x ==  0:
            print(nModel_preds[idx-1])
            print(x)
            print(nModel_preds[idx+1])
    

    all_N_bins = nModel_df[nModel_df['nBin'] == 1].copy()
    nModel_nBins_preds = all_N_bins['pred_nModel'].to_numpy()
    noNModel_nBins_preds = all_N_bins['pred_noNModel'].to_numpy()

    #calculate Stats:
    stat_output = stats.spearmanr(nModel_preds,noNModel_preds)
    r_val = stat_output.statistic
    p_val = stat_output.pvalue

    #Overall Chart
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='col')
    fig.tight_layout(pad = 2)
    plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    axis_ticks = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]


    ax.set_xlabel('Model Trained on N\'s predicting chrom w\o N\'s', fontsize = 30,labelpad= 25)
    ax.set_ylabel('Model Not Traiend on N\'s predicting chrom w\ N\'s', fontsize = 30,labelpad= 12.0)
    #ax[0].spines['top'].set_visible(False)
    #ax[0].spines['right'].set_visible(False)
    #ax[0].spines['bottom'].set_visible(False)
    ax.tick_params(axis = 'x', labelsize = 24, length = 0, pad=15)
    ax.tick_params(axis = 'y', labelsize = 24)
    ax.set_title(the_title + ANIMAL + chrom, y= 1, loc = 'center', fontsize = 32)
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    ax.set_yticks(np.arange(min(axis_ticks), max(axis_ticks), 0.1))
    ax.set_xticks(np.arange(min(axis_ticks), max(axis_ticks), 0.1))
    #ax.plot(axis_ticks, axis_ticks, color = 'r', label = 'perfect_predictions') # axis_ticks used here to create standard diagonal line
    l1 = ax.scatter(nModel_preds,noNModel_preds, s =15, c= 'b', label = 'bins_where_no_n')
    l2 = ax.scatter(nModel_nBins_preds, noNModel_nBins_preds, s =20, c= 'orange',marker ='s' ,label = 'n_bins')
    
    stats_box = (r'$\mathrm{\rho}=%.3f(%.2e)$' % (r_val,p_val, ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.75, 0.05, stats_box, transform=ax.transAxes, fontsize=28, verticalalignment='bottom',horizontalalignment='left', bbox=props)   
    #ax[0].axhline(y=0.5, color = 'black', linewidth = 0.5)
    ax.margins(0)
    ax.grid(linewidth = 0.2)
    plt.show()  

def NoNModelPredictionOnNChrom(chromosome,mse_loss,the_title):

    chrom = getChromAsString(chromosome)

    dataframe1 = pd.read_csv(NoNModel_PREDICTIONS_PATH , sep = ",", header = None, comment = '#')
    dataframe1.columns = ['nBin','target','prediction']

    nBin_df= dataframe1[dataframe1['nBin'] == 1].copy()
    non_df = dataframe1[dataframe1['nBin'] == 0].copy()

    nBin_pred = nBin_df['prediction'].to_numpy()
    nBin_targets = nBin_df['target'].to_numpy()

    nonBin_pred = non_df['prediction'].to_numpy()
    nonBin_targets = non_df['target'].to_numpy()

    #calculate Stats:
    stat_output = stats.spearmanr(dataframe1['target'],dataframe1['prediction'])
    r_val = stat_output.statistic
    p_val = stat_output.pvalue

    #Overall Chart
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='col')
    fig.tight_layout(pad = 2)
    plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    axis_ticks = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]


    ax.set_xlabel('Targets', fontsize = 30,labelpad= 25)
    ax.set_ylabel('Predictions', fontsize = 30,labelpad= 12.0)
    #ax[0].spines['top'].set_visible(False)
    #ax[0].spines['right'].set_visible(False)
    #ax[0].spines['bottom'].set_visible(False)
    ax.tick_params(axis = 'x', labelsize = 24, length = 0, pad=15)
    ax.tick_params(axis = 'y', labelsize = 24)
    ax.set_title(the_title + ANIMAL + chrom, y= 1, loc = 'center', fontsize = 32)
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    ax.set_yticks(np.arange(min(axis_ticks), max(axis_ticks), 0.1))
    ax.set_xticks(np.arange(min(axis_ticks), max(axis_ticks), 0.1))
    ax.plot(axis_ticks, axis_ticks, color = 'r', label = 'perfect_predictions') # axis_ticks used here to create standard diagonal line
    l2 = ax.scatter(nonBin_targets, nonBin_pred, s =15, c= 'b', label = 'bins_where_no_n')
    l1 = ax.scatter(nBin_targets, nBin_pred, s =30, c= 'orange',marker ='s' ,label = 'n_bins')
    
    stats_box = '\n'.join((r'$MSE=%.3f$' % (mse_loss, ),
        r'$\mathrm{\rho}=%.3f(%.2e)$' % (r_val,p_val, ),))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.75, 0.05, stats_box, transform=ax.transAxes, fontsize=28, verticalalignment='bottom',horizontalalignment='left', bbox=props)
    
    
    #ax[0].axhline(y=0.5, color = 'black', linewidth = 0.5)
    ax.margins(0)
    ax.grid(linewidth = 0.2)
    plt.show()

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def buildDatasetFor_NoNModel_NChrom():

    nBins = pd.read_csv(BUILD_PATH, sep = ",", header = 0)
    #test_withN = pd.read_csv(N_PREDICTIONS_PATH2, sep = ",", header = None)
    #test_withN.columns =['prediction','observed']
    test_withoutN = pd.read_csv(N_PREDICTIONS_PATH, sep = ",", header = 0)
    test_withoutN.columns=['prediction','observed']

    n_listT = nBins.values.T.tolist()
    #testWithNList = test_withN.values.T.tolist()
    testWithoutNList =test_withoutN.values.T.tolist()
    

    new_testWithN = []
    new_testWithoutN =[]
    with open(BUILD_OUTPUT_PATH, 'w',newline="") as processedFile:

        writer = csv.writer(processedFile)

        for idx1,expectedOut in enumerate(testWithoutNList[1]):
            
            for idx2,aTrue in enumerate(n_listT[0]):
                
                if str(aTrue)== "nan":
                    continue

                if math.isclose(aTrue,expectedOut, abs_tol=0.000001):
                    new_testWithoutN.append([n_listT[1][idx2],aTrue,testWithoutNList[0][idx1]])
                    n_listT[0][idx2] = -2.1 #this negative value is just a filler, to make sure you don't hit the same aTrue value again
                    break
        

        for row in new_testWithoutN:
            writer.writerow(row)

def CombineNModelDatasets(chromosome):
    
    list_that_breaks_func = ['0.311035901308059','0.494956284761428','0.496092706918716','0.47887310385704','0.48999348282814','0.490538865327835']

    OUTPUT_PATH = './output/NModel_Testing/combined_data.csv'
    noNModel = pd.read_csv(NoNModel_PREDICTIONS_PATH , sep = ",", header = None, comment = '#')
    noNModel.columns = ['nBin','target','prediction']

    nModel= pd.read_csv(NModel_PREDICTIONS_PATH , sep = ",", header = 0, comment = '#')

    noNModel_list = noNModel.values.tolist()
    destructible_noNModel = copy.deepcopy(noNModel_list)
    nModel_list = nModel.values.tolist()

    formatted_nModel = []
    nModel_len = len(nModel_list)
    noNModel_len = len(noNModel_list)
    noNModel_idx = 0
    nModel_idx = 0

    while ((noNModel_idx < noNModel_len) and (nModel_idx < nModel_len)):

        if math.isclose(nModel_list[nModel_idx][1],0.1, abs_tol=0.00001):
            popped = list_that_breaks_func.pop(0)
            formatted_nModel.append([noNModel_list[noNModel_idx][0],noNModel_list[noNModel_idx][1],noNModel_list[noNModel_idx][2],popped])
            nModel_idx+=1
            noNModel_idx+= 1

        elif math.isclose(nModel_list[nModel_idx][1],float(destructible_noNModel[noNModel_idx][1]), abs_tol=0.000001):    
            formatted_nModel.append([noNModel_list[noNModel_idx][0],noNModel_list[noNModel_idx][1],noNModel_list[noNModel_idx][2],nModel_list[noNModel_idx][0]])
            nModel_idx+=1
            noNModel_idx+= 1

        else:
            formatted_nModel.append([noNModel_list[noNModel_idx][0],noNModel_list[noNModel_idx][1],noNModel_list[noNModel_idx][2],'NaN'])
            noNModel_idx+=1

        

    with open(OUTPUT_PATH, 'w', newline="") as outputFile:
        writer = csv.writer(outputFile)

        for row in formatted_nModel:
            writer.writerow(row)    
    
def NModelPredictionOnNoNChrom(chromosome,mse_loss,the_title):

    chrom = getChromAsString(chromosome)

    dataframe1 = pd.read_csv(NModel_PREDICTIONS_PATH , sep = ",", header = 0, comment = '#')


    thePredictions = dataframe1['preds'].to_numpy()
    theTargets = dataframe1['target'].to_numpy()

    #calculate Stats:
    stat_output = stats.spearmanr(theTargets,thePredictions)
    r_val = stat_output.statistic
    p_val = stat_output.pvalue

    print(r_val)
    print(p_val)
    
    #Overall Chart
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='col')
    fig.tight_layout(pad = 2)
    plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    axis_ticks = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]


    ax.set_xlabel('Targets', fontsize = 30,labelpad= 25)
    ax.set_ylabel('Predictions', fontsize = 30,labelpad= 12.0)
    #ax[0].spines['top'].set_visible(False)
    #ax[0].spines['right'].set_visible(False)
    #ax[0].spines['bottom'].set_visible(False)
    ax.tick_params(axis = 'x', labelsize = 24, length = 0, pad=15)
    ax.tick_params(axis = 'y', labelsize = 24)
    ax.set_title(the_title+ ANIMAL + chrom, y= 1, loc = 'center', fontsize = 32)
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    ax.set_yticks(np.arange(min(axis_ticks), max(axis_ticks), 0.1))
    ax.set_xticks(np.arange(min(axis_ticks), max(axis_ticks), 0.1))
    ax.plot(axis_ticks, axis_ticks, color = 'r', label = 'perfect_predictions') # axis_ticks used here to create standard diagonal line
    l1 = ax.scatter(theTargets, thePredictions, s =15, c= 'b', label = 'predictions_vs_targets')

    stats_box = '\n'.join((r'$MSE=%.3f$' % (mse_loss, ),
        r'$\mathrm{\rho}=%.3f(%.2e)$' % (r_val,p_val, ),))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.75, 0.05, stats_box, transform=ax.transAxes, fontsize=28, verticalalignment='bottom',horizontalalignment='left', bbox=props)

    #ax[0].axhline(y=0.5, color = 'black', linewidth = 0.5)
    ax.margins(0)
    ax.grid(linewidth = 0.2)
    plt.show()

def TrainingLossAnalysis():

    for chromosome in range(CHROMOSOMES):
        currentChrom = 'chr' + str(chromosome + 1)
        print('working on: ' + currentChrom)

        if (chromosome+1) < 10:
            chromosome_path = TRAININGLOSS_PATH + '0' + str((chromosome + 1)) + 'TrainLoss.txt'
        else:
            chromosome_path = TRAININGLOSS_PATH + str((chromosome + 1)) + 'TrainLoss.txt'

        loss_df = pd.read_csv(chromosome_path, sep = ",", header = None, comment = '#')
        loss_df.columns = ['Training', 'Validation']
        epochNum = len(loss_df)

        training_loss = loss_df['Training']
        validation_loss = loss_df['Validation']
        max_valloss = round(max(validation_loss) + 0.05,3)
        min_valloss = round(min(validation_loss) - 0.05,3)
        #DEBUG
        #print(training_loss)
        #print(validation_loss)
        epochs = np.arange(1,epochNum + 1,1)

        #Overall Chart
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex='col')
        fig.set_figheight(2) #1080 pixels
        fig.set_figwidth(8) #1920 pixels
        #fig.tight_layout(pad = 2)
        #plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
        #axis_ticks = [0,0.11]
        axis_ticks = [0.015,0.07]


        #ax.set_xlabel('Epochs', fontsize = 26,labelpad= 12)
        #ax.set_ylabel('MSE', fontsize = 26,labelpad= 12.0)
        #ax[0].spines['top'].set_visible(False)
        #ax[0].spines['right'].set_visible(False)
        #ax[0].spines['bottom'].set_visible(False)
        ax.tick_params(axis = 'x', labelsize = 10)
        ax.tick_params(axis = 'y', labelsize = 10)
        #title_string = "Training loss analysis: " + ANIMAL + " " + currentChrom
        ax.set_title(ANIMAL + ": "+ currentChrom, y= 1, loc = 'center', fontsize = 10)
        ax.set_ylim([axis_ticks[0],axis_ticks[1]])
        #ax.set_xlim([0,1])
        ax.set_yticks(np.arange(axis_ticks[0], axis_ticks[1], 0.01))
        #ax.set_xticks(np.arange(min(axis_ticks), max(axis_ticks), 0.1))
        #ax.plot(axis_ticks, axis_ticks, color = 'r', label = 'perfect_predictions') # axis_ticks used here to create standard diagonal line
        l1 = ax.plot(epochs,training_loss,linestyle='-', c= 'b', label = 'Training Loss',linewidth= 2)
        l2 = ax.plot(epochs,validation_loss,linestyle='-', c= 'r', label = 'Validation',linewidth=2)

        #ax[0].axhline(y=0.5, color = 'black', linewidth = 0.5)
        ax.margins(0)
        ax.grid(linewidth = 0.2)
        ax.legend(loc='center',bbox_to_anchor=(0.50, 1.25), prop={'size': 10}, framealpha=0.8, ncol=3, columnspacing = 1)
        plt.savefig( (LOSS_ANALYSIS_PATH + currentChrom +"_loss_analysis_" + ".png" ), dpi = 300, bbox_inches='tight', pad_inches=0.1)
        #plt.show()

def TrainingLossAnalysis_AllChrom():

    all_epochs = []
    all_trainingloss = []
    all_validationloss = []
    subplot_size = 5 

    for chromosome in range(CHROMOSOMES):
        currentChrom = 'chr' + str(chromosome + 1)
        print('working on: ' + currentChrom)

        if (chromosome+1) < 10:
            chromosome_path = TRAININGLOSS_PATH + '0' + str((chromosome + 1)) + 'TrainLoss.txt'
        else:
            chromosome_path = TRAININGLOSS_PATH + str((chromosome + 1)) + 'TrainLoss.txt'

        loss_df = pd.read_csv(chromosome_path, sep = ",", header = None, comment = '#')
        loss_df.columns = ['Training', 'Validation']
        epochNum = len(loss_df)

        training_loss = loss_df['Training']
        validation_loss = loss_df['Validation']
        max_valloss = round(max(validation_loss) + 0.05,3)
        min_valloss = round(min(validation_loss) - 0.05,3)
        #DEBUG
        #print(training_loss)
        #print(validation_loss)
        epochs = np.arange(1,epochNum + 1,1)

        all_epochs.append(epochs)
        all_trainingloss.append(training_loss)
        all_validationloss.append(validation_loss)

    #Overall Chart
    fig, ax = plt.subplots(nrows=subplot_size, ncols=1, sharex='col')
    fig.set_figheight(8) #1080 pixels
    fig.set_figwidth(6) #1920 pixels
    #fig.tight_layout(pad = 2)
    #plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    #axis_ticks = [0,0.11]
    axis_ticks = [0.015,0.07]
    ax[4].set_xlabel('Epochs', fontsize = 20,labelpad= 10)
    ax[2].set_ylabel('MSE', fontsize = 20,labelpad= 10)

    for x in range(subplot_size):
        ax[x].tick_params(axis = 'x', labelsize = 10)
        ax[x].tick_params(axis = 'y', labelsize = 10)
        #title_string = "Training loss analysis: " + ANIMAL + " " + currentChrom
        ax[x].set_title('chr' + str(x+1) +':', y= 0.95, loc = 'center', fontsize = 6)
        ax[x].set_ylim([axis_ticks[0],axis_ticks[1]])
        #ax.set_xlim([0,1])
        ax[x].set_yticks(np.arange(axis_ticks[0], axis_ticks[1], 0.01))
        #ax.set_xticks(np.arange(min(axis_ticks), max(axis_ticks), 0.1))
        #ax.plot(axis_ticks, axis_ticks, color = 'r', label = 'perfect_predictions') # axis_ticks used here to create standard diagonal line
        ax[x].grid(linewidth = 0.2)
        ax[x].plot(all_epochs[x],all_trainingloss[x],linestyle='-', c= 'b', label = 'Training Loss',linewidth= 1.5)
        ax[x].plot(all_epochs[x],all_validationloss[x],linestyle='-', c= 'r', label = 'Validation',linewidth=1.5)

    #ax[0].axhline(y=0.5, color = 'black', linewidth = 0.5)
    #ax.margins(0)
    fig.legend(['Training Loss', 'Validation Loss'], loc = 'center',prop={'size': 10},bbox_to_anchor = [0.50, 0.92],ncol=2) #columnspacing = 1)
    plt.savefig( (LOSS_ANALYSIS_PATH +"allChrom_loss_analysis_" + ".png" ), dpi = 300, bbox_inches='tight', pad_inches=0.1)
    #plt.show()

def TrainingLossAnalysis_WithBars():
    
    first_traininglosses = []
    last_traininglosses = []
    first_validationlosses = []
    last_validationlosses = []

    chrom_list = np.arange(1,CHROMOSOMES+1,1)

    for chromosome in range(CHROMOSOMES):
        currentChrom = 'chr' + str(chromosome + 1)
        print('working on: ' + currentChrom)

        if (chromosome+1) < 10:
            chromosome_path = TRAININGLOSS_PATH + '0' + str((chromosome + 1)) + 'TrainLoss.txt'
        else:
            chromosome_path = TRAININGLOSS_PATH + str((chromosome + 1)) + 'TrainLoss.txt'

        loss_df = pd.read_csv(chromosome_path, sep = ",", header = None, comment = '#')
        loss_df.columns = ['Training', 'Validation']
        epochNum = len(loss_df)

        training_loss = loss_df['Training'].to_list()
        validation_loss = loss_df['Validation'].to_list()

        first_traininglosses.append(mean(training_loss[:3]))
        last_traininglosses.append(mean(training_loss[-3:]))
        first_validationlosses.append(mean(validation_loss[:3]))
        last_validationlosses.append(mean(validation_loss[-3:]))

        

    #DEBUG
    #print(training_loss)
    #print(validation_loss)

    print(last_traininglosses)
    print(last_validationlosses)

    #Overall Chart
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex='col')
    fig.set_figheight(8) #1080 pixels
    fig.set_figwidth(20) #1920 pixels
    #fig.tight_layout(pad = 2)
    #plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    #axis_ticks = [0,0.11]

    if ANIMAL == "Mouse":
        axis_ticks = [0,0.06]
    
    else:
        axis_ticks = [0,0.06]

    bar_width = 0.8
    axes = [0,1]


    ax[1].set_xlabel('Chromosomes', fontsize = 26,labelpad= 12, weight = 'bold')
    ax[1].tick_params(axis = 'x', labelsize = 22)
    
    for axis in axes:
        #ax.set_ylabel('MSE', fontsize = 26,labelpad= 12.0)
        #ax[0].spines['top'].set_visible(False)
        #ax[0].spines['right'].set_visible(False)
        #ax[0].spines['bottom'].set_visible(False)
        ax[axis].tick_params(axis = 'y', labelsize = 22)
        #title_string = "Training loss analysis: " + ANIMAL + " " + currentChrom
        #ax.set_title(title_string, y= 1, loc = 'center', fontsize = 32)
        #ax.set_ylim([0,1])
        ax[axis].set_ylabel('MSE', fontsize = 26,labelpad= 12, weight = 'bold')
        ax[axis].set_xlim([0,max(chrom_list)+1])
        ax[axis].set_ylim([0,axis_ticks[1]])
        ax[axis].set_yticks(np.arange(axis_ticks[0], axis_ticks[1], 0.01))
        ax[axis].set_xticks(np.arange(1, max(chrom_list)+1,1))
        #ax.plot(axis_ticks, axis_ticks, color = 'r', label = 'perfect_predictions') # axis_ticks used here to create standard diagonal line
        #l2 = ax.plot(epochs,validation_loss,linestyle='-', c= 'r', label = 'Validation')

        #ax[0].axhline(y=0.5, color = 'black', linewidth = 0.5)
        ax[axis].margins(0.5)
        ax[axis].grid(linewidth = 0.2, axis = 'y',zorder=0)

    b00 = ax[0].bar(chrom_list,first_traininglosses,color= '#e41a1c', width=bar_width, edgecolor='black', label = 'Starting Training Loss',zorder=3)
    b01 = ax[0].bar(chrom_list,last_traininglosses,color = '#377eb8',width=0.6, edgecolor='black', label = "Ending Training Loss",zorder=4)
    b10 = ax[1].bar(chrom_list,first_validationlosses,color= '#4daf4a', width=bar_width, edgecolor='black', label = 'Starting Validation Loss',zorder=3)
    b11 = ax[1].bar(chrom_list,last_validationlosses,color= '#984ea3', width=0.6, edgecolor='black', label = 'Ending Validation Loss',zorder=4)
    #ax[0].legend(loc='center',bbox_to_anchor=(0.50, 1.05), prop={'size': 20}, framealpha=0.8, ncol=3, columnspacing = 1)
    fig.legend(loc = 'center',prop={'size': 25},bbox_to_anchor = [0.50, 0.98],ncol=2, columnspacing = 1)
    plt.savefig( (LOSS_ANALYSIS_PATH + ANIMAL+"_loss_analysis_bar" + ".png" ), dpi = 300, bbox_inches='tight', pad_inches=0.1)
    #plt.show()

def PlotColoredGCValuesOfPredictions(chrom,highGC_observed,highGC_predicted,midGC_observed,midGC_predicted,lowGC_observed, lowGC_predicted):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.tight_layout(pad = 2)
    plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    fig.set_figheight(11.25) #1080 pixels
    fig.set_figwidth(20) #1920 pixels
    xaxis_ticks = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]
    yaxis_ticks = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]
    
    #x[0].set_xlabel('A-Proportion', fontsize = 26,labelpad= 25)
    ax.set_aspect(1)
    ax.set_ylabel('Predicted CMR', fontsize = 26,labelpad= 25, weight ='bold')
    ax.set_xlabel('Target CMR', fontsize = 26,labelpad= 25, weight = 'bold')
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis = 'x', labelsize = 22, pad=10)
    ax.tick_params(axis = 'y', labelsize = 22, pad=10)


    ax.set_ylim([0.0,1.0])
    ax.set_xlim([0,1])
    ax.set_yticks(np.arange(min(yaxis_ticks), max(yaxis_ticks), 0.1))
    ax.set_xticks(np.arange(min(xaxis_ticks), max(xaxis_ticks), 0.1))

    #0,1,2 = indices for training/testing lists correspond to a threshold ge 0.45, ge 0.4, and everything else
    #l1_0 = ax.scatter(highGC_observed,highGC_predicted, s =20,marker='d', c= 'r', label =( u'$GC-Content \geq 0.45$' + " (Bins: " + str(len(highGC_predicted)) + ")" ),zorder=3)
    #l1_1 = ax.scatter(midGC_observed,midGC_predicted, s =20, c= 'b', label =( u'$GC-Content \geq 0.40$' + " (Bins: " + str(len(midGC_predicted)) + ")" ),zorder=1)
    #l1_2 = ax.scatter(lowGC_observed,lowGC_predicted, s =20,marker='s', c= 'darkorange', label =( u'$GC-Content < 0.40$' + " (Bins: " + str(len(lowGC_predicted)) + ")" ),zorder=2)
    l1_0 = ax.scatter(highGC_observed,highGC_predicted, s =25, c= '#d7191c', label =( u'$GC \geq 0.45$'),zorder=6)
    l1_1 = ax.scatter(midGC_observed,midGC_predicted, s =25, c= '#2c7bb6', label =( u'$ 0.45 > GC \geq 0.40$'),zorder=4)
    l1_2 = ax.scatter(lowGC_observed,lowGC_predicted, s =25, c= '#fdae61', label =( u'$GC < 0.40$'),zorder=5)
    l2 = ax.plot([0.1,0.9],[0.1,0.9], color='black', linestyle = '-',zorder=3, linewidth=3)

    targets = lowGC_observed + midGC_observed + highGC_observed
    predictions = lowGC_predicted + midGC_predicted + highGC_predicted
    rVal_spear, pVal_spear = getSpearman(targets,predictions)
    rVal_pearson, pVal_pearson = getPearson(targets,predictions)

    print("The Spearman rho and pvalue: ",rVal_spear, "(",pVal_spear,")")
    print("The Pearson r and pvalue: ",rVal_pearson, "(",pVal_pearson,")")

    #stats_box = '\n'.join((r'$\mathrm{\rho}=%.3f$' % (rVal),r'$\mathrm{p}{-}\mathrm{value}=%.2e$' % (pVal)))
    stats_spearman = ', '.join((r'Spearman: $\mathrm{\rho}=%.3f$' % (rVal_spear),r'$\mathrm{p}{-}\mathrm{value}=%.2e$' % (pVal_spear)))
    stats_pearson = ', '.join((r'Pearson: $\mathrm{\rho}=%.3f$' % (rVal_pearson),r'$\mathrm{p}{-}\mathrm{value}=%.2e$' % (pVal_pearson)))
    #stats_box = (r"$\mathrm{\rho}=%.3f \t \mathrm{p}{-}\mathrm{value}=%.2e$" % (rVal,pVal))
    props = dict(boxstyle='round', facecolor='white', alpha=0.2)


    ax.grid(linewidth = 0.2,zorder=1)
    ax.legend(loc='center',bbox_to_anchor=(0.50, 1.05), prop={'size': 20}, framealpha=0.8, ncol=4, columnspacing = 1,markerscale=3)
    #ax.text(0.15, 1.09, stats_spearman, transform=ax.transAxes, fontsize=20, verticalalignment='bottom',horizontalalignment='left', bbox=props)
    #ax.text(0.17, 1.14, stats_pearson, transform=ax.transAxes, fontsize=20, verticalalignment='bottom',horizontalalignment='left', bbox=props)
    plt.savefig( (SCATTERPLOTS_SAVE_PATH + chrom + '_targetCMR_vs_PredCMR_' + ANIMAL +".png" ), dpi = 300, bbox_inches='tight', pad_inches=0.1)
    #plt.clf() #close the current figure that's open
    #plt.show()

def getSpearman(array1,array2):
    #calculate Stats:
    stat_output = stats.spearmanr(array1,array2)
    r_val = stat_output.statistic
    p_val = stat_output.pvalue

    return r_val, p_val

def getPearson(array1,array2):
    #calculate:
    stat_output = stats.pearsonr(array1,array2)
    r_val = stat_output.statistic
    p_val = stat_output.pvalue

    return r_val, p_val
    
def PlotRandomPredictions(chrom, highGC_obs, highGC_preds, midGC_obs, midGC_preds, lowGC_obs, lowGC_preds, random_high, random_mid, random_low):

    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
    fig.tight_layout(pad = 2)
    plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    fig.set_figheight(11.25) #1080 pixels
    fig.set_figwidth(20) #1920 pixels
    xaxis_ticks = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]
    yaxis_ticks = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]
    fig.supylabel('Predicted CMR', fontsize = 26) #default is x=0.5 and y= 0.01
    fig.supxlabel("Target CMR",y=0.05,fontsize = 26)
    

    for idx in range(2):

        ax[idx].set_aspect(1)
        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        #ax.spines['bottom'].set_visible(False)
        ax[idx].tick_params(axis = 'x', labelsize = 22, pad=10)
        ax[idx].tick_params(axis = 'y', labelsize = 22, pad=10)
        ax[idx].set_ylim([0.0,1.0])
        ax[idx].set_xlim([0,1])
        ax[idx].set_yticks(np.arange(min(yaxis_ticks), max(yaxis_ticks), 0.1))
        ax[idx].set_xticks(np.arange(min(xaxis_ticks), max(xaxis_ticks), 0.1))
        ax[idx].grid(linewidth = 0.2)

    #0,1,2 = indices for training/testing lists correspond to a threshold ge 0.45, ge 0.4, and everything else
    #l1_0 = ax.scatter(highGC_observed,highGC_predicted, s =20,marker='d', c= 'r', label =( u'$GC-Content \geq 0.45$' + " (Bins: " + str(len(highGC_predicted)) + ")" ),zorder=3)
    #l1_1 = ax.scatter(midGC_observed,midGC_predicted, s =20, c= 'b', label =( u'$GC-Content \geq 0.40$' + " (Bins: " + str(len(midGC_predicted)) + ")" ),zorder=1)
    #l1_2 = ax.scatter(lowGC_observed,lowGC_predicted, s =20,marker='s', c= 'darkorange', label =( u'$GC-Content < 0.40$' + " (Bins: " + str(len(lowGC_predicted)) + ")" ),zorder=2)
    #l1_0 = ax.scatter(highGC_observed,highGC_predicted, s =50,marker='d', c= 'r', label =( u'$GC-Content \geq 0.45$'),zorder=3)
    #l1_1 = ax.scatter(midGC_observed,midGC_predicted, s =50, c= 'b', label =( u'$GC-Content \geq 0.40$'),zorder=1)
    #l1_2 = ax.scatter(lowGC_observed,lowGC_predicted, s =50,marker='s', c= 'darkorange', label =( u'$GC-Content < 0.40$'),zorder=2)
    l0 = ax[0].plot([0.1,0.9],[0.1,0.9], color='black', linestyle = '-',zorder=2, linewidth=2)
    s00= ax[0].scatter(random_high,highGC_preds, s =50,marker='d', c= 'r',zorder=3)
    s01 = ax[0].scatter(random_mid,midGC_preds, s =50, c= 'b',zorder=2)
    s02= ax[0].scatter(random_low,lowGC_preds, s =50,marker='s', c= 'darkorange',zorder=1)
    s13 = ax[1].plot([0.1,0.9],[0.1,0.9], color='black', linestyle = '-', linewidth=2, label= 'Perfect Prediction',zorder=4)
    s10= ax[1].scatter(highGC_obs,highGC_preds, s =50,marker='d', c= 'r', label =( u'$GC \geq 0.45$'),zorder=3)
    s11 = ax[1].scatter(midGC_obs,midGC_preds, s =50, c= 'b', label =( u'$0.45 > GC \geq 0.40$'),zorder=2)
    s12 = ax[1].scatter(lowGC_obs,lowGC_preds, s =50,marker='s', c= 'darkorange', label =( u'$GC < 0.40$'),zorder=1)

    #l4 = ax[idx].scatter(xvalues,all_predictions, s =50,marker='o', c= 'blue', label='test2',zorder=2)
    
    
    #IF YOU WANNA ADD THE RHO and P-VALUE to the actual PLOT:
    #stats_box_random = '\n'.join((r'$MSE=%.3f$' % (0.0, ), r'$\mathrm{\rho}=%.3f$' % (rVal_random),r'$\mathrm{p}{-}\mathrm{value}=%.2e$' % (pVal_random)))
    #props_random = dict(boxstyle='round', facecolor='white', alpha=0.2)
    #stats_box = '\t'.join((r'$\mathrm{\rho}=%.3f$' % (rVal),r'$\mathrm{p}{-}\mathrm{value}=%.2e$' % (pVal)))
    #props = dict(boxstyle='round', facecolor='white', alpha=0.2)


    #ax[0].legend(loc='center',bbox_to_anchor=(0.25, 1.1), prop={'size': 18}, framealpha=0.8)
    #ax[1].legend(loc='center',bbox_to_anchor=(0.25, 1.1), prop={'size': 18}, framealpha=0.8)
    fig.legend(prop={'size': 25},bbox_to_anchor = [0.91, 0.96],ncol=4)
    #ax[0].text(0.6, 1.065, stats_box_random, transform=ax[0].transAxes, fontsize=18, verticalalignment='bottom',horizontalalignment='left', bbox=props_random)
    #ax[1].text(0.6, 1.06, stats_box, transform=ax[1].transAxes, fontsize=18, verticalalignment='bottom',horizontalalignment='left', bbox=props)
    plt.savefig( (SCATTERPLOTS_SAVE_PATH + 'random_preds/' + chrom + '_RandomPreds_vs_ActualPreds_' + ANIMAL +".png" ), dpi = 300, bbox_inches='tight', pad_inches=0.05)
    exit(0)
    #plt.clf() #close the current figure that's open
    #plt.show()

def PlotRandomPredictions_AllChroms(chrom, highGC_obs, highGC_preds, midGC_obs, midGC_preds, lowGC_obs, lowGC_preds, bimodal_high, bimodal_mid, bimodal_low,totally_low, totally_mid, totally_high):
    
    subplot_size = 3
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    fig.tight_layout(pad = 2)
    plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    fig.set_figheight(11) #1080 pixels
    fig.set_figwidth(3) #1920 pixels
    xaxis_ticks = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]
    yaxis_ticks = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]
    plt.xticks(rotation=45)
    #fig.supylabel('Predicted CMR', fontsize = 20) #default is x=0.5 and y= 0.01
    #fig.supxlabel("Target CMR",y=0.05,fontsize = 20)
    marker_size = 2

    for idx in range(subplot_size):

        #ax[idx].set_aspect(1)
        ax[idx].tick_params(axis = 'x', labelsize = 12, pad=2)
        ax[idx].tick_params(axis = 'y', labelsize = 12, pad=10)
        ax[idx].set_ylim([0.0,1.0])
        ax[idx].set_xlim([0,1])
        ax[idx].set_yticks(np.arange(min(yaxis_ticks), max(yaxis_ticks), 0.1))
        ax[idx].set_xticks(np.arange(min(xaxis_ticks), max(xaxis_ticks), 0.1))
        ax[idx].grid(linewidth = 0.2,zorder=0)

    s23 = ax[0].plot([0.1,0.9],[0.1,0.9], color='black', linestyle = '-', linewidth=1,zorder=6)
    s12 = ax[0].scatter(lowGC_obs,totally_low, s =marker_size, c= '#fdae61',zorder=4)
    s11 = ax[0].scatter(midGC_obs,totally_mid, s =marker_size, c= '#2c7bb6',zorder=3)
    s10 = ax[0].scatter(highGC_obs,totally_high, s =marker_size, c= '#d7191c',zorder=5)

    l0 = ax[1].plot([0.1,0.9],[0.1,0.9], color='black', linestyle = '-',zorder=6, linewidth=1)
    s00= ax[1].scatter(bimodal_high,highGC_preds, s =marker_size, c= '#d7191c',zorder=5)
    s01 = ax[1].scatter(bimodal_mid,midGC_preds, s =marker_size, c= '#2c7bb6',zorder=3)
    s02= ax[1].scatter(bimodal_low,lowGC_preds, s =marker_size, c= '#fdae61',zorder=4)

    s13 = ax[2].plot([0.1,0.9],[0.1,0.9], color='black', linestyle = '-', linewidth=1,zorder=6)
    s12 = ax[2].scatter(lowGC_obs,lowGC_preds, s =marker_size, c= '#fdae61', label =( u'$GC < 0.40$'),zorder=4)
    s11 = ax[2].scatter(midGC_obs,midGC_preds, s =marker_size, c= '#2c7bb6', label =( u'$0.45 > GC \geq 0.40$'),zorder=3)
    s10= ax[2].scatter(highGC_obs,highGC_preds, s =marker_size, c= '#d7191c', label =( u'$GC \geq 0.45$'),zorder=5)


    #l4 = ax[idx].scatter(xvalues,all_predictions, s =50,marker='o', c= 'blue', label='test2',zorder=2)
    
    
    #IF YOU WANNA ADD THE RHO and P-VALUE to the actual PLOT:
    #stats_box_random = '\n'.join((r'$MSE=%.3f$' % (0.0, ), r'$\mathrm{\rho}=%.3f$' % (rVal_random),r'$\mathrm{p}{-}\mathrm{value}=%.2e$' % (pVal_random)))
    #props_random = dict(boxstyle='round', facecolor='white', alpha=0.2)
    #stats_box = '\t'.join((r'$\mathrm{\rho}=%.3f$' % (rVal),r'$\mathrm{p}{-}\mathrm{value}=%.2e$' % (pVal)))
    #props = dict(boxstyle='round', facecolor='white', alpha=0.2)


    #ax[0].legend(loc='center',bbox_to_anchor=(0.25, 1.1), prop={'size': 18}, framealpha=0.8)
    #ax[1].legend(loc='center',bbox_to_anchor=(0.25, 1.1), prop={'size': 18}, framealpha=0.8)
    fig.legend(prop={'size': 12},bbox_to_anchor = [1.0, 1.0],ncol=4,markerscale=5)
    #ax[0].text(0.6, 1.065, stats_box_random, transform=ax[0].transAxes, fontsize=18, verticalalignment='bottom',horizontalalignment='left', bbox=props_random)
    #ax[1].text(0.6, 1.06, stats_box, transform=ax[1].transAxes, fontsize=18, verticalalignment='bottom',horizontalalignment='left', bbox=props)
    if chrom == 'chr1':
        plt.savefig( (SCATTERPLOTS_SAVE_PATH + 'random_preds/' + chrom + '_RandomPreds_vs_ActualPreds_' + ANIMAL +".png" ), dpi = 300, bbox_inches='tight', pad_inches=0.05)

    #plt.clf() #close the current figure that's open
    #plt.show()

def RandomPredictions():

    #returns colored predicted and observed vals, mainly used for the neural network 
    gc_df = pd.read_csv(GC_CONTENT_PATH, sep = ",", header = None, comment = '#')
    gc_df.columns = ['Chromosome','startBin', 'endBin','Aprop', 'Bprop','gcContent','atContent']

    random_mse_list = []
    bimodal_mse_list = []
    abcrnet_mse_list = []
  
    for chromosome in range(CHROMOSOMES):
        currentChrom = 'chr' + str(chromosome + 1)
        print('working on: ' + currentChrom)

        if (chromosome+1) < 10:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + '0' + str((chromosome + 1)) + 'TestPredictions.txt'
        else:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + str((chromosome + 1)) + 'TestPredictions.txt'

        predictions_df = pd.read_csv(chromosome_path, sep=',', header=None, comment = '#')
        predictions_df.columns = ['BinID','Predicted', 'Observed']

        gc_chr = gc_df[gc_df['Chromosome'] == currentChrom]
        gcContentList= gc_chr['gcContent'].to_numpy()
        gcTargetList = gc_chr['Aprop'].to_numpy()


        predictionBins = predictions_df['BinID'].to_numpy()
        predictedVals = predictions_df['Predicted'].to_numpy()
        observedVals = predictions_df['Observed'].to_numpy()


        highGC_observed = []
        highGC_predicted = []
        midGC_observed = []
        midGC_predicted = []
        lowGC_observed = []
        lowGC_predicted = []

        for idx,bin in enumerate(predictionBins): 
            '''
            print('********************************')
            print('The predicted value from prediction file is: ',predictedVals[idx])
            print('The observed value from prediction file is: ',observedVals[idx])
            print('The GC-content for bin: ',int(bin)," is ",gcContentList[int(bin)-1])
            print('The target from the gc-content list is: ',gcTargetList[int(bin)-1])
            print('********************************')
            '''
            if gcContentList[int(bin)-1] >= HIGH_GC:
                highGC_observed.append(observedVals[idx])
                highGC_predicted.append(predictedVals[idx])
            
            elif gcContentList[int(bin)-1] >= MID_GC:
                midGC_observed.append(observedVals[idx])
                midGC_predicted.append(predictedVals[idx])
            
            else:
                lowGC_observed.append(observedVals[idx])
                lowGC_predicted.append(predictedVals[idx])
        
        totally_random_preds = np.random.uniform(low=0.1, high =0.9, size = len(predictedVals)).tolist()
        #totally_random_targets = np.random.uniform(low=0.1, high =0.9, size = len(targets)).tolist()

        random_high = np.random.uniform(low=0.1, high =0.9, size = len(highGC_observed)).tolist()
        random_mid = np.random.uniform(low=0.1, high =0.9, size = len(midGC_observed)).tolist()
        random_low = np.random.uniform(low=0.1, high =0.9, size = len(lowGC_observed)).tolist()

        bi_preds = lowGC_predicted + midGC_predicted + highGC_predicted
        birandom_targets = random_low + random_mid + random_high

        trandom_mse = metrics.mean_squared_error(observedVals,totally_random_preds)
        birandom_mse = metrics.mean_squared_error(birandom_targets,bi_preds)
        actual_models_mse = metrics.mean_squared_error(observedVals,predictedVals)
        random_mse_list.append(trandom_mse)
        bimodal_mse_list.append(birandom_mse)
        abcrnet_mse_list.append(actual_models_mse)
        print("     The BiRandom MSE is: ", birandom_mse)
        print("     The Model's MSE is: ",actual_models_mse)

        #PlotRandomPredictions(currentChrom,highGC_observed,highGC_predicted,midGC_observed,midGC_predicted,lowGC_observed,lowGC_predicted, random_high, random_mid, random_low)

        spearman_rho_birandom,spearman_p_birandom = getSpearman(birandom_targets,bi_preds)
        spearman_rho, spearman_p = getSpearman(observedVals,predictedVals)
        print('Spearman: BiRandom Data -- Rho: ',spearman_rho_birandom,' | p-value: ',spearman_p_birandom )
        print('Spearman: Actual Data Data -- Rho: ',spearman_rho,' | p-value: ',spearman_p)

        pearson_r_birandom, pearson_p_birandom = getPearson(birandom_targets,bi_preds)
        pearson_r, pearsonPval = getPearson(observedVals,predictedVals)
        print("***********************************************************")
        print('Pearson: BiRandom Data -- r: ',pearson_r_birandom,' | p-value: ',pearson_p_birandom )
        print('Pearson: Actual Data Data -- r ',pearson_r,' | p-value: ',pearsonPval)
        print("************************************************************")
        print("************************************************************")
        spearman_rho_trandom, spearman_p_trandom = getSpearman(observedVals,totally_random_preds)
        pearson_r_trandom, pearson_p_trandom = getPearson(observedVals,totally_random_preds)
        print("Totally Random Spearman Rho: ",spearman_rho_trandom, " pvalue: ",spearman_p_trandom)
        print("Totally Random Pearson r: ",pearson_r_trandom, " pvalue: ",pearson_p_trandom)
        print("Totally Random MSE: ",trandom_mse)
    
    PlotBimodalRandomMSE_AllChroms(random_mse_list,bimodal_mse_list,abcrnet_mse_list)
        
def RandomPredictions_AllChroms():

    #returns colored predicted and observed vals, mainly used for the neural network 
    gc_df = pd.read_csv(GC_CONTENT_PATH, sep = ",", header = None, comment = '#')
    gc_df.columns = ['Chromosome','startBin', 'endBin','Aprop', 'Bprop','gcContent','atContent']

    random_predictions = []
    random_targets = []
    bimodal_predictions = []
    bimodal_targets = []
    abcrnet_predictions = []
    abcrnet_targets = []
  
    for chromosome in range(CHROMOSOMES):
        currentChrom = 'chr' + str(chromosome + 1)
        print('working on: ' + currentChrom)

        if (chromosome+1) < 10:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + '0' + str((chromosome + 1)) + 'TestPredictions.txt'
        else:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + str((chromosome + 1)) + 'TestPredictions.txt'

        predictions_df = pd.read_csv(chromosome_path, sep=',', header=None, comment = '#')
        predictions_df.columns = ['BinID','Predicted', 'Observed']

        gc_chr = gc_df[gc_df['Chromosome'] == currentChrom]
        gcContentList= gc_chr['gcContent'].to_numpy()
        gcTargetList = gc_chr['Aprop'].to_numpy()


        predictionBins = predictions_df['BinID'].to_numpy()
        predictedVals = predictions_df['Predicted'].to_numpy()
        observedVals = predictions_df['Observed'].to_numpy()


        highGC_observed = []
        highGC_predicted = []
        midGC_observed = []
        midGC_predicted = []
        lowGC_observed = []
        lowGC_predicted = []

        for idx,bin in enumerate(predictionBins): 

            if gcContentList[int(bin)-1] >= HIGH_GC:
                highGC_observed.append(observedVals[idx])
                highGC_predicted.append(predictedVals[idx])
            
            elif gcContentList[int(bin)-1] >= MID_GC:
                midGC_observed.append(observedVals[idx])
                midGC_predicted.append(predictedVals[idx])
            
            else:
                lowGC_observed.append(observedVals[idx])
                lowGC_predicted.append(predictedVals[idx])
        
        totally_random_preds_h = np.random.uniform(low=0, high =1, size = len(highGC_observed)).tolist()
        totally_random_preds_m = np.random.uniform(low=0, high =1, size = len(midGC_observed)).tolist()
        totally_random_preds_l = np.random.uniform(low=0, high =1, size = len(lowGC_observed)).tolist()
        #totally_random_targets = np.random.uniform(low=0.1, high =0.9, size = len(targets)).tolist()
        totally_total_random_preds = totally_random_preds_l + totally_random_preds_m + totally_random_preds_h
        totally_total_random_targs = lowGC_observed + midGC_observed + highGC_observed

        bimodal_high = np.random.uniform(low=0.1, high =0.9, size = len(highGC_observed)).tolist()
        bimodal_mid = np.random.uniform(low=0.1, high =0.9, size = len(midGC_observed)).tolist()
        bimodal_low = np.random.uniform(low=0.1, high =0.9, size = len(lowGC_observed)).tolist()

        bi_preds = lowGC_predicted + midGC_predicted + highGC_predicted
        birandom_targets = bimodal_low + bimodal_mid + bimodal_high

        PlotRandomPredictions_AllChroms(currentChrom, highGC_observed, highGC_predicted, midGC_observed, midGC_predicted, lowGC_observed, lowGC_predicted, bimodal_high, bimodal_mid, bimodal_low,totally_random_preds_l,totally_random_preds_m,totally_random_preds_h)

        #collect the preds and targets for the chromosome and keep in a total list:
        abcrnet_predictions.extend(predictedVals)
        abcrnet_targets.extend(observedVals)

        random_predictions.extend(totally_total_random_preds)
        random_targets.extend(totally_total_random_targs)

        bimodal_predictions.extend(bi_preds)
        bimodal_targets.extend(birandom_targets)

    
    abcrnet_mse = metrics.mean_squared_error(abcrnet_targets,abcrnet_predictions)
    bimodal_mse = metrics.mean_squared_error(bimodal_targets,bimodal_predictions)
    random_mse = metrics.mean_squared_error(random_targets,random_predictions)

    abcrnet_rho, abcrnet_rho_p = getSpearman(abcrnet_targets,abcrnet_predictions)
    bimodal_rho, bimodal_rho_p = getSpearman(bimodal_targets,bimodal_predictions)
    random_rho, random_rho_p = getSpearman(random_targets,random_predictions)

    abcrnet_pr, abcrnet_pr_p = getPearson(abcrnet_targets,abcrnet_predictions)
    bimodal_pr,bimodal_pr_p = getPearson(bimodal_targets,bimodal_predictions)
    random_pr,random_pr_p = getPearson(random_targets,random_predictions)


    print("WARNING! WARNING! WARNING! WARNING! YOU MUST ADJUST TABLE IN RESULTS CHAPTER")
    print('Random:  MSE=',random_mse,' | rho=',random_rho,'(',random_rho_p,') | r=',random_pr,'(',random_pr_p,')')
    print('Bimodal:  MSE=',bimodal_mse,' | rho=',bimodal_rho,'(',bimodal_rho_p,') | r=',bimodal_pr,'(',bimodal_pr_p,')')
    print('ABCRNet:  MSE=',abcrnet_mse,' | rho=',abcrnet_rho,'(',abcrnet_rho_p,') | r=',abcrnet_pr,'(',abcrnet_pr_p,')')

def ColorGCValuesOfPredictions():

    #returns colored predicted and observed vals, mainly used for the neural network 
    gc_df = pd.read_csv(GC_CONTENT_PATH, sep = ",", header = None, comment = '#')
    gc_df.columns = ['Chromosome','startBin', 'endBin','Aprop', 'Bprop','gcContent','atContent']
    all_low = []
    all_mid = []
    all_high = []

  
    for chromosome in range(CHROMOSOMES):
        currentChrom = 'chr' + str(chromosome + 1)
        print('working on: ' + currentChrom)

        if (chromosome+1) < 10:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + '0' + str((chromosome + 1)) + 'TestPredictions.txt'
        else:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + str((chromosome + 1)) + 'TestPredictions.txt'

        predictions_df = pd.read_csv(chromosome_path, sep=',', header=None, comment = '#')
        predictions_df.columns = ['BinID','Predicted', 'Observed']

        gc_chr = gc_df[gc_df['Chromosome'] == currentChrom]
        gcContentList= gc_chr['gcContent'].to_numpy()
        gcTargetList = gc_chr['Aprop'].to_numpy()


        predictionBins = predictions_df['BinID'].to_numpy()
        predictedVals = predictions_df['Predicted'].to_numpy()
        observedVals = predictions_df['Observed'].to_numpy()

        highGC_observed = []
        highGC_predicted = []
        midGC_observed = []
        midGC_predicted = []
        lowGC_observed = []
        lowGC_predicted = []

        for idx,bin in enumerate(predictionBins): 
            '''
            print('********************************')
            print('The predicted value from prediction file is: ',predictedVals[idx])
            print('The observed value from prediction file is: ',observedVals[idx])
            print('The GC-content for bin: ',int(bin)," is ",gcContentList[int(bin)-1])
            print('The target from the gc-content list is: ',gcTargetList[int(bin)-1])
            print('********************************')
            '''
            if gcContentList[int(bin)-1] >= HIGH_GC:
                highGC_observed.append(observedVals[idx])
                highGC_predicted.append(predictedVals[idx])
            
            elif gcContentList[int(bin)-1] >= MID_GC:
                midGC_observed.append(observedVals[idx])
                midGC_predicted.append(predictedVals[idx])
            
            else:
                lowGC_observed.append(observedVals[idx])
                lowGC_predicted.append(predictedVals[idx])


        #BIN COUNTS INSPECTION:
        if currentChrom == 'chr19':    

            #specific_bins = [x for x in lowGC_predicted if x < 0.35]
            bin_count = 0

            for idx,x in enumerate(lowGC_observed):
                #if x < 0.15:
                    #if lowGC_predicted[idx] >= 0.1 and lowGC_predicted[idx] < 0.35:
                bin_count += 1

            print("TOTAL LOW count: ",len(lowGC_observed))
            print('RAW count: ',bin_count)
            print('As a %: ', (round((bin_count/len(lowGC_observed)*100),2)) )
            print('=============================================================')

            bin_count = 0
            for idx,x in enumerate(midGC_observed):
                if x < 0.3 or x > 0.7:
                    #if midGC_predicted[idx] >= 0.4 and midGC_predicted[idx] < 0.8:
                    bin_count += 1
            
            print("TOTAL MID count: ",len(midGC_observed))            
            print('RAW count: ',bin_count)
            print('As a %: ', (round((bin_count/len(midGC_observed)*100),2)) )
            print('=============================================================')

            bin_count = 0
            for idx,x in enumerate(highGC_observed):
                if x < 0.6:
                    #if highGC_observed[idx] < 0.3:
                    bin_count += 1
            
            print("TOTAL HIGH count: ",len(highGC_observed))            
            print('RAW count: ',bin_count)
            print('As a %: ', (round((bin_count/len(highGC_observed)*100),2)) )
            print('=============================================================')
            print ("Total alltogether: ", (len(lowGC_observed) + len(midGC_observed) + len(highGC_observed)))
        
        all_low.append(len(lowGC_observed))
        all_mid.append(len(midGC_observed))
        all_high.append(len(highGC_observed))
        
        #BIN COUNTS INSPECTION END


        PlotColoredGCValuesOfPredictions(currentChrom,highGC_observed,highGC_predicted,midGC_observed,midGC_predicted,lowGC_observed,lowGC_predicted)

    print('average count of low bins across genome: ', mean(all_low))
    print('average count of mid bins across genome: ', mean(all_mid))
    print('average count of high bins across genome: ', mean(all_high))

def plotTestingLosses(loss):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.tight_layout(pad = 2)
    plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    fig.set_figheight(11.25) #1080 pixels
    fig.set_figwidth(20) #1920 pixels
    xaxis_values = np.arange(1,(CHROMOSOMES + 1), 1)
    #yaxis_ticks = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]
    
    #ax.set_aspect(1./ax.get_data_ratio()) #in order to make it a square, the x and y coordinates need to be same length, otherwise it won't work, so use this method to get the right ratio
    ax.set_ylabel('MSE', fontsize = 26,labelpad= 25, weight = 'bold')
    ax.set_xlabel('Chromosomes', fontsize = 26,labelpad= 25, weight = 'bold')
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis = 'x', labelsize = 22, pad=10)
    ax.tick_params(axis = 'y', labelsize = 22, pad=10)

    #title_string = "Testing Loss for " + ANIMAL + " Chromosomes"
    #ax.set_title(title_string, y= 1, loc = 'center', fontsize = 28,pad=20)
    #ax.set_ylim([0.0,1.0])
    #ax.set_xlim([0,1])

    if ANIMAL == "Mouse":
        ax.set_yticks(np.arange(0,0.06, 0.005))
        ax.set_xticks(np.arange(min(xaxis_values), max(xaxis_values)+1, 1))
    else: #assume human
        ax.set_yticks(np.arange(0,0.16, 0.01))
        ax.set_xticks(np.arange(min(xaxis_values), max(xaxis_values)+1, 1))        
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        #loss_copy = loss.copy()
        #loss_copy.pop(18) #index 18 is chromosome 19, lets get rid of this to get the average loss without this particular chromosome
        #second_mean_loss = mean(loss_copy)
        #ax.hlines(y=second_mean_loss,xmin=xaxis_values[0],xmax=xaxis_values[-1], label= ("Mean Loss w/o chr19 (" + str(round(second_mean_loss,3)) + ")"), color='g',linewidth=3,alpha=1)



    l1_0 = ax.bar(xaxis_values, loss, color= 'r')
    mean_loss = mean(loss)
    ax.hlines(y=mean_loss,xmin=xaxis_values[0],xmax=xaxis_values[-1], label= ("Mean (" + str(round(mean_loss,3)) + ")"), color='b',linewidth=3,alpha=1)

    ax.grid(linewidth = 0.2)
    ax.legend(prop={'size': 25})
    plt.savefig( (ALLCHROMCHARTS_PATH + 'TestingLosses_' + ANIMAL +".png" ), dpi = 300, bbox_inches='tight', pad_inches=0.1)
    #plt.clf() #close the current figure that's open
    #plt.show()

def TestingLossAnalysis():

    testing_loss = []
    for chromosome in range(CHROMOSOMES):
        currentChrom = 'chr' + str(chromosome + 1)

        if (chromosome+1) < 10:
            log_path = TRAININGLOG_PATH+ '0' + str((chromosome + 1)) + 'TrainingLog.txt'
        else:
            log_path = TRAININGLOG_PATH + str((chromosome + 1)) + 'TrainingLog.txt'
    
        with open(log_path) as logFile:

            for line in logFile:
                if "Testing losses" in line:
                    first_split = line.split("tensor(")
                    second_split = first_split[1].split(",")
                    loss_value = float(second_split[0])
                    print(currentChrom, ' Testing Loss: ', loss_value)
                    testing_loss.append(loss_value)
                    break;
    
    print("The median testing loss is: ", statistics.median(testing_loss))
    print("The Average MSE: ",mean(testing_loss))

    plotTestingLosses(testing_loss)

def TargetCMR_GC_Distribution(chromosomes):
    #returns colored predicted and observed vals, mainly used for the neural network 
    gc_df = pd.read_csv(GC_CONTENT_PATH, sep = ",", header = None, comment = '#')
    gc_df.columns = ['Chromosome','startBin', 'endBin','Aprop', 'Bprop','gcContent','atContent']

    lowGC = []
    midGC = []
    highGC = []
  
    for chromosome in range(chromosomes):
        currentChrom = 'chr' + str(chromosome + 1)
        print('working on: ' + currentChrom)

        if (chromosome+1) < 10:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + '0' + str((chromosome + 1)) + 'TestPredictions.txt'
        else:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + str((chromosome + 1)) + 'TestPredictions.txt'

        predictions_df = pd.read_csv(chromosome_path, sep=',', header=None, comment = '#')
        predictions_df.columns = ['BinID','Predicted', 'Observed']

        gc_chr = gc_df[gc_df['Chromosome'] == currentChrom]
        gcContentList= gc_chr['gcContent'].to_numpy()
        gcTargetList = gc_chr['Aprop'].to_numpy()


        predictionBins = predictions_df['BinID'].to_numpy()
        predictedVals = predictions_df['Predicted'].to_numpy()
        observedVals = predictions_df['Observed'].to_numpy()

        for idx,bin in enumerate(predictionBins): 
            '''
            print('********************************')
            print('The predicted value from prediction file is: ',predictedVals[idx])
            print('The observed value from prediction file is: ',observedVals[idx])
            print('The GC-content for bin: ',int(bin)," is ",gcContentList[int(bin)-1])
            print('The target from the gc-content list is: ',gcTargetList[int(bin)-1])
            print('********************************')
            '''
            if gcContentList[int(bin)-1] >= HIGH_GC:
                highGC.append(observedVals[idx])
            
            elif gcContentList[int(bin)-1] >= MID_GC:
                midGC.append(observedVals[idx])
            
            else:
                lowGC.append(observedVals[idx])
    
    plotTargetCMR_GC_Distribution(highGC,midGC,lowGC)

def plotTargetCMR_GC_Distribution(high,mid,low):

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='col')
    #fig.tight_layout(pad = 2)
    fig.set_figheight(11.25) #1080 pixels
    fig.set_figwidth(20) #1920 pixels
    #plt.subplots_adjust(wspace = 0.2, hspace = 0.4)

    ax.set_title("Target CMR Range Based on GC-Content for "+ ANIMAL, y= 1, loc = 'center', fontsize = 28)
    ax.set_ylabel('Target CMR', fontsize = 26,labelpad= 25.0)
    ax.set_xlabel('GC-Content', fontsize = 26,labelpad= 25.0)
    bp1 = ax.boxplot([low,mid,high], labels =['Low','Mid','High'],patch_artist=True,notch=True)

    #fill with colors:
    colors = ['darkorange','b','r']
    for patch, color in zip(bp1['boxes'],colors):
        patch.set_facecolor(color)
    
    #change median color
    for median in bp1['medians']:
        median.set_color('black')


    ax.tick_params(axis = 'both', labelsize = 24)
    #ax.set_yticks(np.arange(0,1.1, 0.1))
    #plt.grid(axis = 'y', linewidth = 0.2)
    #plt.show()
    plt.savefig( (ALLCHROMCHARTS_PATH + ANIMAL +"_GCvsTargetRange_" + ".png" ), dpi = 300)

def BestWorstTrainingLoss2Charts():
    if ANIMAL == "Human":
        dsName = 'hg38:'
        best_worst_chroms = [0,10] #we -1 because of the indexing below
        testing_loss = [0.0276,0.0555]
        epochs=[]
    else:
        dsName = 'mm10:'
        best_worst_chroms = [10,13] #we -1 because of the indexing below
        testing_loss = [0.034,0.015]
        epochs = []
    
    training_losses = []
    validation_losses = []
    chroms = []

    for chromosome in best_worst_chroms:
        currentChrom = 'chr' + str(chromosome + 1)
        chroms.append(currentChrom)
        print('working on: ' + currentChrom)

        if (chromosome+1) < 10:
            chromosome_path = TRAININGLOSS_PATH + '0' + str((chromosome + 1)) + 'TrainLoss.txt'
        else:
            chromosome_path = TRAININGLOSS_PATH + str((chromosome + 1)) + 'TrainLoss.txt'

        loss_df = pd.read_csv(chromosome_path, sep = ",", header = None, comment = '#')
        loss_df.columns = ['Training', 'Validation']
        epochNum = len(loss_df)

        training_losses.append(loss_df['Training'])
        validation_losses.append(loss_df['Validation'])

        #DEBUG
        #print(training_loss)
        #print(validation_loss)
        epochs.append(np.arange(1,epochNum + 1,1))

    #Overall Chart
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex='col')
    fig.set_figheight(11.25) #1080 pixels
    fig.set_figwidth(20) #1920 pixels
    #fig.tight_layout(pad=1)
    #plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    axis_ticks = [0,0.06]


    ax[1].set_xlabel('Epochs', fontsize = 26,labelpad= 6)
    ax[0].set_ylabel('Loss (MSE)', fontsize = 26,labelpad= 12.0)
    ax[1].set_ylabel('Loss (MSE)', fontsize = 26,labelpad= 12.0)
    #ax[0].spines['top'].set_visible(False)
    #ax[0].spines['right'].set_visible(False)
    #ax[0].spines['bottom'].set_visible(False)
    ax[0].tick_params(axis = 'x', labelsize = 22)
    ax[0].tick_params(axis = 'y', labelsize = 22)
    ax[1].tick_params(axis = 'x', labelsize = 22)
    ax[1].tick_params(axis = 'y', labelsize = 22)
    #title_string = "Training loss analysis: " + ANIMAL + " " + currentChrom
    ax[0].set_title(dsName + chroms[0] , y= 1, loc = 'center', fontsize = 26)
    ax[1].set_title(dsName + chroms[1], y= 1, loc = 'center', fontsize = 26)
    ax[0].set_ylim([0,0.06])
    ax[1].set_ylim([0,0.06])
    #ax.set_xlim([0,1])
    ax[0].set_yticks(np.arange(min(axis_ticks), max(axis_ticks), 0.005))
    ax[1].set_yticks(np.arange(min(axis_ticks), max(axis_ticks), 0.005))
    #ax.set_xticks(np.arange(min(axis_ticks), max(axis_ticks), 0.1))
    #ax.plot(axis_ticks, axis_ticks, color = 'r', label = 'perfect_predictions') # axis_ticks used here to create standard diagonal line
    l1 = ax[0].plot(epochs[0],training_losses[0],linestyle='-', c= 'b', label = 'Training Loss')
    l2 = ax[0].plot(epochs[0],validation_losses[0],linestyle='-', c= 'r', label = 'Validation Loss')
    l3 = ax[1].plot(epochs[1],training_losses[1],linestyle='-', c= 'b', label = 'Training Loss')
    l4 = ax[1].plot(epochs[1],validation_losses[1],linestyle='-', c= 'r', label = 'Validation Loss')
    
    l5 = ax[0].axhline(y=testing_loss[0],color='purple', label='Testing Loss')
    l6 = ax[1].axhline(y=testing_loss[1],color='purple')

    #ax[0].axhline(y=0.5, color = 'black', linewidth = 0.5)
    ax[0].margins(0)
    ax[1].margins(0)
    ax[0].grid(linewidth = 0.2)
    ax[1].grid(linewidth = 0.2)
    ax[0].legend(prop={'size':18})
    plt.savefig( (BESTWORST_LOSS_PATH + ANIMAL +"_BestWorstTrainingLoss_" + ".png" ), dpi = 300, bbox_inches='tight', pad_inches=0)
    #plt.show()

def BestWorstTrainingLoss3Charts():
    if ANIMAL == "Human":
        dsName = 'hg38:'
        best_worst_chroms = [4,6,9] #we -1 because of the indexing below
        epochs=[]
    else:
        dsName = 'mm10:'
        best_worst_chroms = [10,13] #we -1 because of the indexing below
        testing_loss = [0.034,0.015]
        epochs = []
    
    training_losses = []
    validation_losses = []
    chroms = []

    for chromosome in best_worst_chroms:
        currentChrom = 'chr' + str(chromosome + 1)
        chroms.append(currentChrom)
        print('working on: ' + currentChrom)

        if (chromosome+1) < 10:
            chromosome_path = TRAININGLOSS_PATH + '0' + str((chromosome + 1)) + 'TrainLoss.txt'
        else:
            chromosome_path = TRAININGLOSS_PATH + str((chromosome + 1)) + 'TrainLoss.txt'

        loss_df = pd.read_csv(chromosome_path, sep = ",", header = None, comment = '#')
        loss_df.columns = ['Training', 'Validation']
        epochNum = len(loss_df)

        training_losses.append(loss_df['Training'])
        validation_losses.append(loss_df['Validation'])

        #DEBUG
        #print(training_loss)
        #print(validation_loss)
        epochs.append(np.arange(1,epochNum + 1,1))

    #Overall Chart
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex='col')
    fig.set_figheight(11.25) #1080 pixels
    fig.set_figwidth(20) #1920 pixels
    #fig.tight_layout(pad=5)
    #plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    axis_ticks = [0.02,0.06]
    
    ax[2].set_xlabel('Epochs', fontsize = 26,labelpad= 6)

    for subplot in range(len(best_worst_chroms)):

        ax[subplot].set_ylabel('Loss (MSE)', fontsize = 26,labelpad= 12.0)
        #ax[0].spines['top'].set_visible(False)
        #ax[0].spines['right'].set_visible(False)
        #ax[0].spines['bottom'].set_visible(False)
        ax[subplot].tick_params(axis = 'x', labelsize = 22)
        ax[subplot].tick_params(axis = 'y', labelsize = 22)
        #title_string = "Training loss analysis: " + ANIMAL + " " + currentChrom
        ax[subplot].set_title(dsName + chroms[subplot] , y= 1, loc = 'center', fontsize = 26)
        ax[subplot].set_ylim([0.02,0.06])
        #ax.set_xlim([0,1])
        ax[subplot].set_yticks(np.arange(min(axis_ticks), max(axis_ticks), 0.005))
        #ax.set_xticks(np.arange(min(axis_ticks), max(axis_ticks), 0.1))
        #ax.plot(axis_ticks, axis_ticks, color = 'r', label = 'perfect_predictions') # axis_ticks used here to create standard diagonal line
        l1 = ax[subplot].plot(epochs[subplot],training_losses[subplot],linestyle='-', c= 'b', label = 'Training Loss')
        l2 = ax[subplot].plot(epochs[subplot],validation_losses[subplot],linestyle='-', c= 'r', label = 'Validation Loss')
        
        #l5 = ax[subplot].axhline(y=testing_loss[subplot],color='purple', label='Testing Loss')
        #l6 = ax[subplot].axhline(y=testing_loss[subplot],color='purple')

        #ax[0].axhline(y=0.5, color = 'black', linewidth = 0.5)
        ax[subplot].margins(0)
        ax[subplot].grid(linewidth = 0.2)
    
    ax[0].legend(prop={'size':18})
    #plt.savefig( (BESTWORST_LOSS_PATH + ANIMAL +"_BestWorstTrainingLoss_" + ".png" ), dpi = 300)
    plt.show()

def CMR_GC_Histograms():
    gc_df = pd.read_csv(GC_CONTENT_PATH, sep = ",", header = None, comment = '#')
    gc_df.columns = ['Chromosome','startBin', 'endBin','Aprop', 'Bprop','gcContent','atContent']

    lowGC = []
    midGC = []
    highGC = []

    for chromosome in range(CHROMOSOMES):
        currentChrom = 'chr' + str(chromosome + 1)
        print('working on: ' + currentChrom)

        if (chromosome+1) < 10:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + '0' + str((chromosome + 1)) + 'TestPredictions.txt'
        else:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + str((chromosome + 1)) + 'TestPredictions.txt'

        predictions_df = pd.read_csv(chromosome_path, sep=',', header=None, comment = '#')
        predictions_df.columns = ['BinID','Predicted', 'Observed']

        gc_chr = gc_df[gc_df['Chromosome'] == currentChrom]
        gcContentList= gc_chr['gcContent'].to_numpy()
        gcTargetList = gc_chr['Aprop'].to_numpy()


        predictionBins = predictions_df['BinID'].to_numpy()
        predictedVals = predictions_df['Predicted'].to_numpy()
        observedVals = predictions_df['Observed'].to_numpy()

        for idx,bin in enumerate(predictionBins): 
            '''
            print('********************************')
            print('The predicted value from prediction file is: ',predictedVals[idx])
            print('The observed value from prediction file is: ',observedVals[idx])
            print('The GC-content for bin: ',int(bin)," is ",gcContentList[int(bin)-1])
            print('The target from the gc-content list is: ',gcTargetList[int(bin)-1])
            print('********************************')
            '''
            if gcContentList[int(bin)-1] >= HIGH_GC:
                highGC.append(observedVals[idx])
            
            elif gcContentList[int(bin)-1] >= MID_GC:
                midGC.append(observedVals[idx])
            
            else:
                lowGC.append(observedVals[idx])

    plotCMR_GC_Histograms(highGC,midGC,lowGC)

def CMR_Stats():
    gc_df = pd.read_csv(GC_CONTENT_PATH, sep = ",", header = None, comment = '#')
    gc_df.columns = ['Chromosome','startBin', 'endBin','Aprop', 'Bprop','gcContent','atContent']

    high_a2b_flipped = 0
    high_b2a_flipped = 0
    mid_a2b_flipped = 0
    mid_b2a_flipped = 0
    low_a2b_flipped = 0
    low_b2a_flipped = 0
    total_high_bins = 0
    total_mid_bins = 0
    total_low_bins = 0

    for chromosome in range(CHROMOSOMES):
        currentChrom = 'chr' + str(chromosome + 1)
        print('working on: ' + currentChrom)

        if (chromosome+1) < 10:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + '0' + str((chromosome + 1)) + 'TestPredictions.txt'
        else:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + str((chromosome + 1)) + 'TestPredictions.txt'

        predictions_df = pd.read_csv(chromosome_path, sep=',', header=None, comment = '#')
        predictions_df.columns = ['BinID','Predicted', 'Observed']

        gc_chr = gc_df[gc_df['Chromosome'] == currentChrom]
        gcContentList= gc_chr['gcContent'].to_numpy()
        gcTargetList = gc_chr['Aprop'].to_numpy()


        predictionBins = predictions_df['BinID'].to_numpy()
        predictedVals = predictions_df['Predicted'].to_numpy()
        observedVals = predictions_df['Observed'].to_numpy()

        for idx,bin in enumerate(predictionBins): 

            if gcContentList[int(bin)-1] >= HIGH_GC: #high
                total_high_bins += 1

                if predictedVals[idx] < 0.5: 

                    if observedVals[idx] > 0.5:
                        high_a2b_flipped += 1
                
                elif predictedVals[idx] > 0.5:

                    if observedVals[idx] < 0.5:
                        high_b2a_flipped += 1
                else:
                    raise Exception("Found a value of 0.5, please check")

            
            elif gcContentList[int(bin)-1] >= MID_GC: #mid
                total_mid_bins += 1

                if predictedVals[idx] < 0.5: 

                    if observedVals[idx] > 0.5:
                        mid_a2b_flipped += 1
                
                elif predictedVals[idx] > 0.5:

                    if observedVals[idx] < 0.5:
                        mid_b2a_flipped += 1
                else:
                    raise Exception("Found a value of 0.5, please check")

            else:                                   #low
                total_low_bins += 1

                if predictedVals[idx] < 0.5: 

                    if observedVals[idx] > 0.5:
                        low_a2b_flipped += 1
                
                elif predictedVals[idx] > 0.5:

                    if observedVals[idx] < 0.5:
                        low_b2a_flipped += 1
                
                else:
                    raise Exception("Found a value of 0.5, please check")
    
    print("Low GC class, a2b flipped: ", round((low_a2b_flipped/total_low_bins)*100,1), "% | b2a flipped: ", round((low_b2a_flipped/total_low_bins)*100,1), '%')
    print("Mid GC class,  a2b flipped: ", round((mid_a2b_flipped/total_mid_bins)*100,1), "% | b2a flipped: ", round((mid_b2a_flipped/total_mid_bins)*100,1), '%')
    print("High GC class, a2b flipped: ", round((high_a2b_flipped/total_high_bins)*100,1), "% | b2a flipped: ", round((high_b2a_flipped/total_high_bins)*100,1), '%')
            
def plotCMR_GC_Histograms(high,mid,low):

    #lets calc some stats:
    h_Median = statistics.median(high)
    h_mean = mean(high)
    h_std = statistics.pstdev(high)
    m_Median = statistics.median(mid)
    m_mean = mean(mid)
    m_std = statistics.pstdev(mid)
    l_Median = statistics.median(low)
    l_mean = mean(low)
    l_std = statistics.pstdev(low)

    q1_l = np.percentile(low,25)
    q3_l = np.percentile(low,75)
    q1_m = np.percentile(mid,25)
    q3_m = np.percentile(mid,75)
    q1_h = np.percentile(high,25)
    q3_h = np.percentile(high,75)

    print('**************************************')
    print("MEAN HIGH-GC: ",h_mean)
    print("MEAN MID-GC: ", m_mean)
    print("MEAN LOW-GC: ", l_mean)
    print("MEDIAN HIGH-GC: ", h_Median)
    print("MEDIAN MID-GC: ", m_Median)
    print("MEDIAN LOW-GC: ", l_Median)

    print('STD HIGH-GC: ',h_std)
    print('STD MID-GC: ', m_std)
    print('STD LOW-GC: ', l_std)
    print('**************************************')
    print("The total count of high-GC: ",len(high), " %: ", len(high) / (len(high) + len(mid) + len(low)))
    print("The total count of mid-GC: ",len(mid), " %: ", len(mid) / (len(high) + len(mid) + len(low)))
    print("The total count of low-GC: ",len(low), " %: ", len(low) / (len(high) + len(mid) + len(low)))

    # create 99% confidence interval 
    low_ci = scipy.stats.t.interval(alpha=0.99, df=len(low)-1, loc=np.mean(low),  scale=scipy.stats.sem(low))
    mid_ci = scipy.stats.t.interval(alpha=0.99, df=len(mid)-1, loc=np.mean(mid),  scale=scipy.stats.sem(mid)) 
    high_ci = scipy.stats.t.interval(alpha=0.99, df=len(high)-1, loc=np.mean(high),  scale=scipy.stats.sem(high)) 

    print('The low GC mean CI: ',low_ci)
    print("The mid GC mean CI: ",mid_ci)
    print('The high GC mean CI: ',high_ci)

    print('****************************')
    print("The Q1 LOW-GC: ", q1_l)
    print("The Q3 LOW-GC: ",q3_l)
    print("The Q1 MID-GC: ", q1_m)
    print("The Q3 MID-GC: ",q3_m)
    print("The Q1 HIGH-GC: ", q1_h)
    print("The Q3 HIGH-GC: ",q3_h)


    fig, ax = plt.subplots(nrows=3, ncols=1, sharex='col')
    #fig.tight_layout(pad = 2)
    fig.set_figheight(11.25) #1080 pixels
    fig.set_figwidth(20) #1920 pixels
    #plt.subplots_adjust(wspace = 0.2, hspace = 0.4)
    binz = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    yticks1 = np.arange(0,4250,250)
    yticks2 = np.arange(0,3500,500)
    ax[0].set_yticks(yticks2) 
    ax[1].set_yticks(yticks2) 
    ax[2].set_yticks(yticks2)
    
    #ax.set_title("Target CMR Range Based on GC-Content for "+ ANIMAL, y= 1, loc = 'center', fontsize = 28)
    ax[2].set_xticks(binz)
    ax[2].set_xlabel('Target CMR', fontsize = 26,labelpad= 25, weight='bold')
    hp0 = ax[0].hist(high, bins=binz, alpha=1, color='#d7191c',edgecolor = 'black',label=(u'$GC \geq 0.45$'),zorder=3)
    vl0 = ax[0].axvline(h_Median, color='#abd9e9',linestyle='solid', linewidth=2,zorder=4)
    #vl01 = ax[0].axvline(h_std, color='fuchsia', linestyle=':', linewidth=2)
    hp1 = ax[1].hist(mid, bins=binz, alpha=1, color='#2c7bb6',edgecolor = 'black',label=(u'$ 0.45 > GC \geq 0.40$'),zorder=3)
    vl1 = ax[1].axvline(m_Median, color='#abd9e9', linestyle='solid', linewidth=2,zorder=4)
    #vl11 = ax[1].axvline(m_std, color='fuchsia', linestyle=':', linewidth=2)
    hp2 = ax[2].hist(low, bins=binz, alpha=1, color='#fdae61',edgecolor = 'black',label=(u'$GC < 0.40$'),zorder=3)
    vl2 = ax[2].axvline(l_Median, color='#abd9e9', linestyle='solid', linewidth=2, label = 'Median',zorder=4)
    #vl21 = ax[2].axvline(l_std, color='fuchsia', linestyle=':', linewidth=2, label= 'SD')
    
    for subplot in range(3):
        ax[subplot].tick_params(axis = 'both', labelsize = 22)
        ax[subplot].grid(linewidth = 0.2,zorder=0)
        ax[subplot].set_ylim([min(yticks2),max(yticks2)])
        #ax[subplot].margins(y=0.5)

    ax[1].set_ylabel('Count', fontsize = 26,labelpad= 25.0, weight='bold')
    #ax.set_yticks(np.arange(0,1.1, 0.1))
    #plt.grid(axis = 'y', linewidth = 0.2)
    #plt.show()
    #fig.legend(handles,labels, loc='center')
    fig.legend(prop={'size': 22},bbox_to_anchor = [0.82, 0.95],ncol=4)
    plt.savefig( (ALLCHROMCHARTS_PATH + ANIMAL +"_GCvsCMR_HISTOGRAMS_" + ".png" ), dpi = 300, bbox_inches='tight', pad_inches=0)

def AnalyzeGCviaBoxPlot():
    gc_df_m = pd.read_csv('./data/mouse/mouse_gc_content.csv', sep = ",", header = None, comment = '#')
    gc_df_h = pd.read_csv('./data/human/human_gc_content.csv', sep = ",", header = None, comment = '#')
    gc_df_m.columns = ['Chromosome','startBin', 'endBin','Aprop', 'Bprop','gcContent','atContent']
    gc_df_h.columns = ['Chromosome','startBin', 'endBin','Aprop', 'Bprop','gcContent','atContent']

    gc_df_m_cleaned = gc_df_m.drop(gc_df_m[(gc_df_m['gcContent'] == 0.0) & (gc_df_m['atContent'] == 0.0)].index) #need to provide the index
    gc_df_h_cleaned = gc_df_h.drop(gc_df_h[(gc_df_h['gcContent'] == 0.0) & (gc_df_h['atContent'] == 0.0)].index) #need to provide the index
    

    all_gc_m = gc_df_m_cleaned['gcContent'].to_numpy()
    all_gc_h = gc_df_h_cleaned['gcContent'].to_numpy()

    print('Here are the statistics of the GC-Content for: Mouse')
    print('min: ',min(all_gc_m))
    print('max: ',max(all_gc_m))
    print('Average: ',mean(all_gc_m))
    print('Median:', statistics.median(all_gc_m))
    q1,q2,q3 = np.percentile(all_gc_m,[25,50,75])
    print('The precentile results are: Q1: ',q1)
    print('The precentile results are: Q2: ',q2)
    print('The precentile results are: Q3: ',q3)
    print('\n')

    print('Here are the statistics of the GC-Content for: Human')
    print('min: ',min(all_gc_h))
    print('max: ',max(all_gc_h))
    print('Average: ',mean(all_gc_h))
    print('Median:', statistics.median(all_gc_h))
    q1,q2,q3 = np.percentile(all_gc_h,[25,50,75])
    print('The precentile results are: Q1: ',q1)
    print('The precentile results are: Q2: ',q2)
    print('The precentile results are: Q3: ',q3)
    print('\n')

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='col')
    #fig.tight_layout(pad = 2)
    fig.set_figheight(11.25) #1080 pixels
    fig.set_figwidth(20) #1920 pixels
    #plt.subplots_adjust(wspace = 0.2, hspace = 0.4)
    yticks = np.arange(0.2,0.8,0.025)
    ax.set_yticks(yticks)
    ax.set_ylim([0.3,0.7])

    #ax.set_title("Boxplots of GC-Content for "+ ANIMAL, y= 1, loc = 'center', fontsize = 28)
    ax.set_ylabel('GC-Content', fontsize = 30,labelpad= 12.0, weight='bold')
    bp1 = ax.boxplot([all_gc_m,all_gc_h], labels = ['Mouse','Human'], patch_artist=True)
    ax.set_xticklabels(labels=["Mouse","Human"], weight='bold',fontsize = 30)
    ax.tick_params(axis = 'both', labelsize = 22)
    plt.grid(axis = 'y', linewidth = 0.2)
    
    #fill with colors:
    colors = ['navy','maroon']
    for patch, color in zip(bp1['boxes'],colors):
        patch.set_facecolor(color)
    
    #change median color
    for median in bp1['medians']:
        median.set_color('yellow')


    #plt.show()
    plt.savefig( ('./output/hm_output/GCAnalyzed_HM_' + ".png" ), dpi = 300, bbox_inches='tight', pad_inches=0)

def Target_GC_Correlation_Analysis():
    print('Conducting Analysis on the Correlation between CMR and GC-Content:')
    gc_df = pd.read_csv(GC_CONTENT_PATH, sep = ",", header = None, comment = '#')
    gc_df.columns = ['Chromosome','startBin', 'endBin','Aprop', 'Bprop','gcContent','atContent']
    gc_df.dropna(inplace=True)

    gcContentList= gc_df['gcContent'].to_numpy()
    gcTargetList = gc_df['Aprop'].to_numpy()

    rVal_spear, pVal_spear = getSpearman(gcContentList,gcTargetList)
    rVal_pearson, pVal_pearson = getPearson(gcContentList,gcTargetList)
    print('Spearman -- Rho: ',rVal_spear,' | p-value: ',pVal_spear)
    print('Pearson -- Rho: ',rVal_pearson,' | p-value: ',pVal_pearson)

    #The above does the correlation between the target CMR and the GC-content but what
    # if we want to look  whether the predicted CMR is correlated with the GC-Content?
    # right now your other colored chart only checks if the predicted CMR is correlated with the Target CMR, not anything to do with GC-Content

def getValidGCValues(bins,gc_array):

    valid_gc = []
    for bin in bins:
        valid_gc.append(gc_array[int(bin-1)])
    
    return np.array(valid_gc)

def Predictions_GC_Correlation_Analysis():
    #returns colored predicted and observed vals, mainly used for the neural network 
    gc_df = pd.read_csv(GC_CONTENT_PATH, sep = ",", header = None, comment = '#')
    gc_df.columns = ['Chromosome','startBin', 'endBin','Aprop', 'Bprop','gcContent','atContent']

    total_targets = []
    total_gc = []

  
    for chromosome in range(CHROMOSOMES):
        currentChrom = 'chr' + str(chromosome + 1)
        print('working on: ' + currentChrom)

        if (chromosome+1) < 10:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + '0' + str((chromosome + 1)) + 'TestPredictions.txt'
        else:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + str((chromosome + 1)) + 'TestPredictions.txt'

        predictions_df = pd.read_csv(chromosome_path, sep=',', header=None, comment = '#')
        predictions_df.columns = ['BinID','Predicted', 'Observed']

        gc_chr = gc_df[gc_df['Chromosome'] == currentChrom]
        gcContentList= gc_chr['gcContent'].to_numpy()
        gcTargetList = gc_chr['Aprop'].to_numpy()


        predictionBins = predictions_df['BinID'].to_numpy()
        predictedVals = predictions_df['Predicted'].to_numpy()
        observedVals = predictions_df['Observed'].to_numpy()

        validGC_values = getValidGCValues(predictionBins,gcContentList) #here we get the GC content for the predicted bins only 

        #TESTING to check if gc and preds match:
        #for idx,val in enumerate(predictedVals):
        #    print('The predicted value is: ',val, ' and the GC value is: ',validGC_values[idx])
        
        spearman_rho, spearman_p = getSpearman(predictedVals,validGC_values)
        pearson_r, pearson_p = getPearson(predictedVals,validGC_values)

        print('     The Spearman rho = ', spearman_rho, ' and p = ',spearman_p)
        print('     The Pearson r    = ', pearson_r, '    and p = ', pearson_p)

        total_gc.extend(validGC_values)
        total_targets.extend(observedVals)
    
    
    spearman_rho, spearman_p = getSpearman(total_targets,total_gc)
    pearson_r, pearson_p = getPearson(total_targets,total_gc)

    print('TOTAL GC VS OBSERVED: Spearman rho = ', spearman_rho, ' and p = ',spearman_p)
    print('TOTAL GC VS OBSERVED: Pearson r    = ', pearson_r, '    and p = ', pearson_p)

def Predictions_OverOrUnderEstimated():
    #returns colored predicted and observed vals, mainly used for the neural network 
    gc_df = pd.read_csv(GC_CONTENT_PATH, sep = ",", header = None, comment = '#')
    gc_df.columns = ['Chromosome','startBin', 'endBin','Aprop', 'Bprop','gcContent','atContent']
    
    total_overE_low = []
    total_overE_mid = []
    total_overE_high = []
    total_underE_low = []
    total_underE_mid = []
    total_underE_high = []
    total_overall_predictions = 0

    over_all_chrom= []
    under_all_chrom=[]
  
    for chromosome in range(CHROMOSOMES):
        currentChrom = 'chr' + str(chromosome + 1)
        print('working on: ' + currentChrom)

        if (chromosome+1) < 10:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + '0' + str((chromosome + 1)) + 'TestPredictions.txt'
        else:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + str((chromosome + 1)) + 'TestPredictions.txt'

        predictions_df = pd.read_csv(chromosome_path, sep=',', header=None, comment = '#')
        predictions_df.columns = ['BinID','Predicted', 'Observed']

        gc_chr = gc_df[gc_df['Chromosome'] == currentChrom]
        gcContentList= gc_chr['gcContent'].to_numpy()
        gcTargetList = gc_chr['Aprop'].to_numpy()


        predictionBins = predictions_df['BinID'].to_numpy()
        predictedVals = predictions_df['Predicted'].to_numpy()
        observedVals = predictions_df['Observed'].to_numpy()

        validGC_values = getValidGCValues(predictionBins,gcContentList) #here we get the GC content for the predicted bins only 

        overE_low = []
        overE_mid = []
        overE_high = []
        underE_low = []
        underE_mid = []
        underE_high = []
        total_chrom_predictions = len(predictedVals)
        total_overall_predictions += total_chrom_predictions

        for idx,pred in enumerate(predictedVals):
            abs_diff = abs(observedVals[idx] - pred)

            if abs_diff > 0.1:

                if pred > observedVals[idx]:
                    if validGC_values[idx] >= 0.45:
                        overE_high.append(pred)
                        total_overE_high.append(pred)
                    elif validGC_values[idx] >= 0.4:
                        overE_mid.append(pred)
                        total_overE_mid.append(pred)
                    else:
                        overE_low.append(pred)
                        total_overE_low.append(pred)

                else:
                    if validGC_values[idx] >= 0.45:
                        underE_high.append(pred)
                        total_underE_high.append(pred)
                    elif validGC_values[idx] >= 0.4:
                        underE_mid.append(pred)
                        total_underE_mid.append(pred)
                    else:
                        underE_low.append(pred)
                        total_underE_low.append(pred)
        
        over_low = round(len(overE_low)/total_chrom_predictions,2)
        over_mid = round(len(overE_mid)/total_chrom_predictions,2)
        over_high = round(len(overE_high)/total_chrom_predictions,2)
        under_low = round(len(underE_low)/total_chrom_predictions,2)
        under_mid = round(len(underE_mid)/total_chrom_predictions,2)
        under_high = round(len(underE_high)/total_chrom_predictions,2)
        print("     Total Predictions: ",total_chrom_predictions)
        print("     Over-estimated predictions low GC: ",over_low )
        print("     Over-estimated predictions mid GC: ",over_mid )
        print("     Over-estimated predictions high GC: ",over_high )
        print("     *****************************************************************************************")
        print("     Under-estimated predictions low GC: ",under_low)
        print("     Under-estimated predictions mid GC: ",under_mid)
        print("     Under-estimated predictions high GC: ",under_high )

        #capture per chrom
        over_all_chrom.append([over_low,over_mid,over_high])
        under_all_chrom.append([under_low,under_mid,under_high])


    print("         TOTAL - Over-estimated predictions low GC: ", round(len(total_overE_low)/total_overall_predictions,2))
    print("         TOTAL - Over-estimated predictions mid GC: ", round(len(total_overE_mid)/total_overall_predictions,2))
    print("         TOTAL - Over-estimated predictions high GC: ", round(len(total_overE_high)/total_overall_predictions,2))
    print("         *****************************************************************************************")
    print("         TOTAL - Under-estimated predictions low GC: ", round(len(total_underE_low)/total_overall_predictions,2))
    print("         TOTAL - Under-estimated predictions mid GC: ", round(len(total_underE_mid)/total_overall_predictions,2))
    print("         TOTAL - Under-estimated predictions high GC: ", round(len(total_underE_high)/total_overall_predictions,2))

    PlotOverorUnderEstiamted(over_all_chrom,under_all_chrom)

def PlotOverorUnderEstiamted(over,under):
    
    chrom_list = np.arange(1,CHROMOSOMES+1,1)
    
    #Overall Chart
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='col')
    fig.set_figheight(20) #1080 pixels
    fig.set_figwidth(10) #1920 pixels
    #fig.tight_layout(pad = 2)
    #plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    #axis_ticks = [0,0.11]
    
    if ANIMAL == 'Mouse':
        xticks = [-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4]
        xtick_labels = [40,30,20,10,0,10,20,30,40]
        ax.set_xlim([-0.42,0.42])
    else:
        xticks = [-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5]
        xtick_labels = [50,40,30,20,10,0,10,20,30,40,50]
        ax.set_xlim([-0.52,0.52])


    ax.set_xlabel('Under (%)                    Over (%)', fontsize = 20,labelpad= 12)
    ax.tick_params(axis = 'x', labelsize = 22)
    

    #ax.set_ylabel('MSE', fontsize = 26,labelpad= 12.0)
    #ax[0].spines['top'].set_visible(False)
    #ax[0].spines['right'].set_visible(False)
    #ax[0].spines['bottom'].set_visible(False)
    #ax.tick_params(axis = 'y', labelsize = 22)
    #title_string = "Training loss analysis: " + ANIMAL + " " + currentChrom
    #ax.set_title(title_string, y= 1, loc = 'center', fontsize = 32)
    #ax.set_ylim([0,1])
    ax.set_ylabel('Chromosomes', fontsize = 26,labelpad= 12)
    #ax.set_ylim([0,axis_ticks[1]])
    ax.set_yticks(chrom_list)
    ax.set_ylim([0.5,max(chrom_list)+0.5])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    #ax.set_yticks(chrom_list)
    #ax.plot(axis_ticks, axis_ticks, color = 'r', label = 'perfect_predictions') # axis_ticks used here to create standard diagonal line
    #l2 = ax.plot(epochs,validation_loss,linestyle='-', c= 'r', label = 'Validation')

    #ax[0].axhline(y=0.5, color = 'black', linewidth = 0.5)
    ax.margins(0.5)
    ax.grid(linewidth = 0.2, axis = 'both',zorder=0)
    
    plt.axvline(x = 0, color = 'black')

    bar_width = 0.2
    for idx,bin in enumerate(over):

        ax.barh(chrom_list[idx] - 0.2,bin[0],color= 'darkorange', height=bar_width)
        ax.barh(chrom_list[idx],bin[1],color= 'blue', height = bar_width)
        ax.barh(chrom_list[idx] + 0.2,bin[2],color= 'red', height = bar_width)

    for idx,bin in enumerate(under):

        ax.barh(chrom_list[idx] - 0.2,-bin[0],color= 'darkorange', height=bar_width)
        ax.barh(chrom_list[idx],-bin[1],color= 'blue', height = bar_width)
        ax.barh(chrom_list[idx] + 0.2,-bin[2],color= 'red', height = bar_width)
        
    
    #b01 = ax.bar(chrom_list,last_traininglosses,color = 'green',width=0.6,label = "Ending Training Loss",zorder=4)
    #b10 = ax.bar(chrom_list,first_validationlosses,color= 'blue', width=bar_width,label = 'Starting Validation Loss',zorder=3)
    #b11 = ax.bar(chrom_list,last_validationlosses,color= 'darkorange', width=0.6,label = 'Ending Validation Loss',zorder=4)
    #ax[0].legend(loc='center',bbox_to_anchor=(0.50, 1.05), prop={'size': 20}, framealpha=0.8, ncol=3, columnspacing = 1)
    fig.legend(loc = 'center',prop={'size': 25},bbox_to_anchor = [0.50, 0.98],ncol=2, columnspacing = 1)
    plt.savefig( (OVERUNDER_PATH + ANIMAL+"_overunder_analysis_bar" + ".png" ), dpi = 300, bbox_inches='tight', pad_inches=0.1)
    #plt.show()

def PlotAll_CMRinOne():
    #returns colored predicted and observed vals, mainly used for the neural network 
    gc_df = pd.read_csv(GC_CONTENT_PATH, sep = ",", header = None, comment = '#')
    gc_df.columns = ['Chromosome','startBin', 'endBin','Aprop', 'Bprop','gcContent','atContent']
    
    highGC_observed = []
    highGC_predicted = []
    midGC_observed = []
    midGC_predicted = []
    lowGC_observed = []
    lowGC_predicted = []

  
    for chromosome in range(CHROMOSOMES):
        currentChrom = 'chr' + str(chromosome + 1)
        print('working on: ' + currentChrom)

        if (chromosome+1) < 10:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + '0' + str((chromosome + 1)) + 'TestPredictions.txt'
        else:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + str((chromosome + 1)) + 'TestPredictions.txt'

        predictions_df = pd.read_csv(chromosome_path, sep=',', header=None, comment = '#')
        predictions_df.columns = ['BinID','Predicted', 'Observed']

        gc_chr = gc_df[gc_df['Chromosome'] == currentChrom]
        gcContentList= gc_chr['gcContent'].to_numpy()
        gcTargetList = gc_chr['Aprop'].to_numpy()


        predictionBins = predictions_df['BinID'].to_numpy()
        predictedVals = predictions_df['Predicted'].to_numpy()
        observedVals = predictions_df['Observed'].to_numpy()


        for idx,bin in enumerate(predictionBins): 
            '''
            print('********************************')
            print('The predicted value from prediction file is: ',predictedVals[idx])
            print('The observed value from prediction file is: ',observedVals[idx])
            print('The GC-content for bin: ',int(bin)," is ",gcContentList[int(bin)-1])
            print('The target from the gc-content list is: ',gcTargetList[int(bin)-1])
            print('********************************')
            '''
            if gcContentList[int(bin)-1] >= HIGH_GC:
                highGC_observed.append(observedVals[idx])
                highGC_predicted.append(predictedVals[idx])
            
            elif gcContentList[int(bin)-1] >= MID_GC:
                midGC_observed.append(observedVals[idx])
                midGC_predicted.append(predictedVals[idx])
            
            else:
                lowGC_observed.append(observedVals[idx])
                lowGC_predicted.append(predictedVals[idx])

    PlotColoredGCValuesOfPredictions('All_CHROMOSOMES',highGC_observed,highGC_predicted,midGC_observed,midGC_predicted,lowGC_observed,lowGC_predicted)

def Plot_GC_VS_CMR():

    gc_mouse_path = './data/mouse/mouse_gc_content.csv'
    gc_human_path = './data/human/human_gc_content.csv'

    #mouse:
    mouse_gc_df = pd.read_csv(gc_mouse_path, sep = ",", header = None, comment = '#')
    mouse_gc_df.columns = ['Chromosome','startBin', 'endBin','Aprop', 'Bprop','gcContent','atContent']

    #mouse_gc_df = mouse_org_gc_df[mouse_org_gc_df['Chromosome'] == 'chr1'].copy()
    mouse_gc_df.dropna(inplace=True) #remove any of the nan from the gc content file (in the current chromosome)
    m_gcContentList= mouse_gc_df['gcContent'].to_numpy()
    m_gcTargetList = mouse_gc_df['Aprop'].to_numpy()

    #human:
    human_gc_df = pd.read_csv(gc_human_path, sep = ",", header = None, comment = '#')
    human_gc_df.columns = ['Chromosome','startBin', 'endBin','Aprop', 'Bprop','gcContent','atContent']

    #human_gc_df = human_org_gc_df[human_org_gc_df['Chromosome'] == 'chr1'].copy()
    human_gc_df.dropna(inplace=True) #remove any of the nan from the gc content file (in the current chromosome)
    h_gcContentList= human_gc_df['gcContent'].to_numpy()
    h_gcTargetList = human_gc_df['Aprop'].to_numpy()


    #STATS:
    rVal_spear, pVal_spear = getSpearman(m_gcTargetList,m_gcContentList)
    rVal_pearson, pVal_pearson = getPearson(m_gcTargetList,m_gcContentList)
    print("MOUSE: The Spearman rho and pvalue: ",rVal_spear, "(",pVal_spear,")")
    print("MOUSE: The Pearson r and pvalue: ",rVal_pearson, "(",pVal_pearson,")")
    
    print("*****")
    rVal_spear2, pVal_spear2 = getSpearman(h_gcTargetList,h_gcContentList)
    rVal_pearson2, pVal_pearson2 = getPearson(h_gcTargetList,h_gcContentList)
    print("HUMAN: The Spearman rho and pvalue: ",rVal_spear2, "(",pVal_spear2,")")
    print("HUMAN: The Pearson r and pvalue: ",rVal_pearson2, "(",pVal_pearson2,")")

    #Graph
    fig, ax = plt.subplots(nrows=2, ncols=1,sharex ='col')
    fig.tight_layout(pad = 2)
    plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    fig.set_figheight(11.25) #1080 pixels
    fig.set_figwidth(20) #1920 pixels
    xaxis_ticks = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]
    #yaxis_ticks = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]
    yaxis_ticks0 = [0.3,0.4,0.5,0.57]
    yaxis_ticks1 = [0.3,0.4,0.5,0.6,0.67]
    ax[0].set_yticks(np.arange(min(yaxis_ticks0), max(yaxis_ticks0), 0.05))
    ax[0].set_ylim([min(yaxis_ticks0),max(yaxis_ticks0)])
    ax[1].set_yticks(np.arange(min(yaxis_ticks1), max(yaxis_ticks1), 0.05))
    ax[1].set_ylim([min(yaxis_ticks1),max(yaxis_ticks1)])
    ax[1].set_xlabel('Target CMR', fontsize = 26,labelpad= 25,weight='bold')

    for x in range(0,2):

        ax[x].set_ylabel('GC-Content', fontsize = 26,labelpad= 25,weight='bold')
        ax[x].tick_params(axis = 'x', labelsize = 22, pad=10)
        ax[x].tick_params(axis = 'y', labelsize = 22, pad=10)
        ax[x].set_xlim([0,1])
        ax[x].set_xticks(np.arange(min(xaxis_ticks), max(xaxis_ticks), 0.1))
        ax[x].grid(linewidth = 0.2)
    
    l1_0 = ax[0].scatter(m_gcTargetList,m_gcContentList, s =25,marker='o',label = "Mouse", c= '#7b3294',zorder=3)
    l1_0 = ax[1].scatter(h_gcTargetList,h_gcContentList, s =25,marker='o',label = "Human", c= '#008837',zorder=3)

    fig.legend(loc = 'center',prop={'size': 28},bbox_to_anchor = [0.50, 0.98],ncol=2, columnspacing = 1,markerscale=5)

    plt.savefig( (ALL_OUTPUT+'_targetCMR_vs_GC_hm'+".png" ), dpi = 300, bbox_inches='tight', pad_inches=0.1)

def PlotLinRegVsABCRNet():
    print('This plots the Linear Regression Model vs. ABCRNet one Chromosome at a time:')
    #FIRST GET ABCRNET:
    gc_df = pd.read_csv(GC_CONTENT_PATH, sep = ",", header = None, comment = '#')
    gc_df.columns = ['Chromosome','startBin', 'endBin','Aprop', 'Bprop','gcContent','atContent']
    linreg_df = pd.read_csv(LINREG_DATA_PATH, sep=',',header=0)

  
    for chromosome in range(CHROMOSOMES):
        currentChrom = 'chr' + str(chromosome + 1)
        print('working on: ' + currentChrom)

        if (chromosome+1) < 10:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + '0' + str((chromosome + 1)) + 'TestPredictions.txt'
        else:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + str((chromosome + 1)) + 'TestPredictions.txt'

        predictions_df = pd.read_csv(chromosome_path, sep=',', header=None, comment = '#')
        predictions_df.columns = ['BinID','Predicted', 'Observed']

        gc_chr = gc_df[gc_df['Chromosome'] == currentChrom]
        gcContentList= gc_chr['gcContent'].to_numpy()
        gcTargetList = gc_chr['Aprop'].to_numpy()


        predictionBins = predictions_df['BinID'].to_numpy()
        predictedVals = predictions_df['Predicted'].to_numpy()
        observedVals = predictions_df['Observed'].to_numpy()


        highGC_observed = []
        highGC_predicted = []
        midGC_observed = []
        midGC_predicted = []
        lowGC_observed = []
        lowGC_predicted = []

        for idx,bin in enumerate(predictionBins): 
            '''
            print('********************************')
            print('The predicted value from prediction file is: ',predictedVals[idx])
            print('The observed value from prediction file is: ',observedVals[idx])
            print('The GC-content for bin: ',int(bin)," is ",gcContentList[int(bin)-1])
            print('The target from the gc-content list is: ',gcTargetList[int(bin)-1])
            print('********************************')
            '''
            if gcContentList[int(bin)-1] >= HIGH_GC:
                highGC_observed.append(observedVals[idx])
                highGC_predicted.append(predictedVals[idx])
            
            elif gcContentList[int(bin)-1] >= MID_GC:
                midGC_observed.append(observedVals[idx])
                midGC_predicted.append(predictedVals[idx])
            
            else:
                lowGC_observed.append(observedVals[idx])
                lowGC_predicted.append(predictedVals[idx])
        
        #SETUP linreg:
        current_lingreg = linreg_df[linreg_df['Chromosome'] == currentChrom]

        linreg_highGC = current_lingreg[current_lingreg['GC'] =='high']
        linreg_highGC_preds = linreg_highGC['Predictions'].to_numpy()
        linreg_highGC_targets = linreg_highGC['Targets'].to_numpy()

        linreg_midGC = current_lingreg[current_lingreg['GC'] =='mid']
        linreg_midGC_preds = linreg_midGC['Predictions'].to_numpy()
        linreg_midGC_targets = linreg_midGC['Targets'].to_numpy()

        linreg_lowGC = current_lingreg[current_lingreg['GC'] =='low']
        linreg_lowGC_preds = linreg_lowGC['Predictions'].to_numpy()
        linreg_lowGC_targets = linreg_lowGC['Targets'].to_numpy()

        #PLOT ABCRNET VS LINREG for CHROM:

        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
        fig.tight_layout(pad = 2)
        plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
        fig.set_figheight(11.25) #1080 pixels
        fig.set_figwidth(20) #1920 pixels
        xaxis_ticks = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]
        yaxis_ticks = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]
        fig.supylabel('Predicted CMR', fontsize = 24, weight='bold') #default is x=0.5 and y= 0.01
        fig.supxlabel("Target CMR",y=0.05,fontsize = 24, weight='bold')
        

        for idx in range(2):

            ax[idx].set_aspect(1)
            #ax.spines['top'].set_visible(False)
            #ax.spines['right'].set_visible(False)
            #ax.spines['bottom'].set_visible(False)
            ax[idx].tick_params(axis = 'x', labelsize = 22, pad=10)
            ax[idx].tick_params(axis = 'y', labelsize = 22, pad=10)
            ax[idx].set_ylim([0.0,1.0])
            ax[idx].set_xlim([0,1])
            ax[idx].set_yticks(np.arange(min(yaxis_ticks), max(yaxis_ticks), 0.1))
            ax[idx].set_xticks(np.arange(min(xaxis_ticks), max(xaxis_ticks), 0.1))
            ax[idx].grid(linewidth = 0.2)

        if ANIMAL == 'Mouse':
            title_string = 'Mouse: ' + currentChrom
        else:
            title_string = 'Human: ' + currentChrom

        plt.title(title_string, x=-0.05, y=1, loc = 'center', fontsize = 18,pad=10)
        l0 = ax[0].plot([0.1,0.9],[0.1,0.9], color='black', linestyle = '-',zorder=2, linewidth=2)
        s00= ax[0].scatter(linreg_highGC_targets,linreg_highGC_preds, s =25, c= '#d7191c',zorder=3)
        s01 = ax[0].scatter(linreg_midGC_targets,linreg_midGC_preds, s =25, c= '#2c7bb6',zorder=2)
        s02= ax[0].scatter(linreg_lowGC_targets,linreg_lowGC_preds, s =25, c= '#fdae61',zorder=1)
        s13 = ax[1].plot([0.1,0.9],[0.1,0.9], color='black', linestyle = '-', linewidth=2,zorder=4)
        s10= ax[1].scatter(highGC_observed,highGC_predicted, s =25, c= '#d7191c', label =( u'$GC \geq 0.45$'),zorder=3)
        s11 = ax[1].scatter(midGC_observed,midGC_predicted, s =25, c= '#2c7bb6', label =( u'$0.45 > GC \geq 0.40$'),zorder=2)
        s12 = ax[1].scatter(lowGC_observed,lowGC_predicted, s =25, c= '#fdae61', label =( u'$GC < 0.40$'),zorder=1)

        fig.legend(prop={'size': 25},bbox_to_anchor = [0.75, 0.98],ncol=4,markerscale=5)
        plt.savefig( (SCATTERPLOTS_SAVE_PATH + 'linreg_vs_ABCRNet/' + currentChrom + '_linreg_vs_abcrnet_' + ANIMAL +".png" ), dpi = 300, bbox_inches='tight', pad_inches=0.05)
        exit(0)
        #plt.clf() #close the current figure that's open
        #plt.show()

def PlotBimodalRandomMSE_AllChroms(random, bimodal, abcrnet):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.tight_layout(pad = 2)
    plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    fig.set_figheight(11.25) #1080 pixels
    fig.set_figwidth(20) #1920 pixels
    xaxis_values = np.arange(1,(CHROMOSOMES + 1), 1)
    #yaxis_ticks = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]
    
    #ax.set_aspect(1./ax.get_data_ratio()) #in order to make it a square, the x and y coordinates need to be same length, otherwise it won't work, so use this method to get the right ratio
    ax.set_ylabel('MSE', fontsize = 26,labelpad= 25)
    ax.set_xlabel('Chromosomes', fontsize = 26,labelpad= 25)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis = 'x', labelsize = 22, pad=10)
    ax.tick_params(axis = 'y', labelsize = 22, pad=10)

    #title_string = "Perdiction Error on " + ANIMAL + " Genome"
    #ax.set_title(title_string, y= 1, loc = 'center', fontsize = 28,pad=20)
    ylim = 0.23
    ax.set_ylim([0.0,ylim])
    #ax.set_xlim([0,1])

    #if ANIMAL == "Mouse":
        #ax.set_yticks(np.arange(0,0.06, 0.005))
        #ax.set_xticks(np.arange(min(xaxis_values), max(xaxis_values)+1, 1))
    #else: #assume human
        #ax.set_yticks(np.arange(0,0.16, 0.01))
        #ax.set_xticks(np.arange(min(xaxis_values), max(xaxis_values)+1, 1))
        #loss_copy = loss.copy()
        #loss_copy.pop(18) #index 18 is chromosome 19, lets get rid of this to get the average loss without this particular chromosome
        #second_mean_loss = mean(loss_copy)
        #ax.hlines(y=second_mean_loss,xmin=xaxis_values[0],xmax=xaxis_values[-1], label= ("Mean Loss w/o chr19 (" + str(round(second_mean_loss,3)) + ")"), color='g',linewidth=3,alpha=1)

    ax.set_yticks(np.arange(0,ylim, 0.01))
    ax.set_xticks(np.arange(min(xaxis_values), max(xaxis_values)+1, 1))

    bar_width = 0.3
    l1_0 = ax.bar(xaxis_values-bar_width,random,width=bar_width, color= 'cyan',edgecolor = 'black',align='center', label='Random',zorder=3)
    l2_0 = ax.bar(xaxis_values,bimodal,width=bar_width,color='lime',edgecolor = 'black',align='center', label = 'Bimodal Random',zorder=3)
    l3_0 = ax.bar(xaxis_values+bar_width,abcrnet,width=bar_width,color='fuchsia',edgecolor = 'black',align='center', label ='ABCRNet',zorder=3)
    #mean_loss = mean(loss)
    #ax.hlines(y=mean_loss,xmin=xaxis_values[0],xmax=xaxis_values[-1], label= ("Mean (" + str(round(mean_loss,3)) + ")"), color='b',linewidth=3,alpha=1)

    ax.grid(linewidth = 0.2,axis='y')
    ax.margins(0)
    fig.legend(loc = 'center',prop={'size': 25},bbox_to_anchor = [0.50, 0.98],ncol=3) #columnspacing = 1
    plt.savefig( (ALLCHROMCHARTS_PATH + 'RANDOM_MODELS_MSE_COMPARISON_' + ANIMAL +".png" ), dpi = 300, bbox_inches='tight', pad_inches=0.1)
    #plt.clf() #close the current figure that's open
    #plt.show()    

def Deviations_by_GC():

    #returns colored predicted and observed vals, mainly used for the neural network 
    gc_df = pd.read_csv(GC_CONTENT_PATH, sep = ",", header = None, comment = '#')
    gc_df.columns = ['Chromosome','startBin', 'endBin','Aprop', 'Bprop','gcContent','atContent']
    
    #Preds
    all_low_CMR_p = []
    all_mid_CMR_p = []
    all_high_CMR_p = []

    #Targets
    all_low_CMR_t = []
    all_mid_CMR_t = []
    all_high_CMR_t = []

    #IN THIS SECTION GET THE MEAN CMR PER GC-CONTENT BIN:
    for chromosome in range(CHROMOSOMES):
        currentChrom = 'chr' + str(chromosome + 1)
        print('working on: ' + currentChrom)

        if (chromosome+1) < 10:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + '0' + str((chromosome + 1)) + 'TestPredictions.txt'
        else:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + str((chromosome + 1)) + 'TestPredictions.txt'

        predictions_df = pd.read_csv(chromosome_path, sep=',', header=None, comment = '#')
        predictions_df.columns = ['BinID','Predicted', 'Observed']

        gc_chr = gc_df[gc_df['Chromosome'] == currentChrom]
        gcContentList= gc_chr['gcContent'].to_numpy()
        gcTargetList = gc_chr['Aprop'].to_numpy()


        predictionBins = predictions_df['BinID'].to_numpy()
        predictedVals = predictions_df['Predicted'].to_numpy()
        observedVals = predictions_df['Observed'].to_numpy()

        highGC_observed = []
        highGC_predicted = []
        midGC_observed = []
        midGC_predicted = []
        lowGC_observed = []
        lowGC_predicted = []

        for idx,bin in enumerate(predictionBins): 

            if gcContentList[int(bin)-1] >= HIGH_GC:
                highGC_observed.append(observedVals[idx])
                highGC_predicted.append(predictedVals[idx])
            
            elif gcContentList[int(bin)-1] >= MID_GC:
                midGC_observed.append(observedVals[idx])
                midGC_predicted.append(predictedVals[idx])
            
            else:
                lowGC_observed.append(observedVals[idx])
                lowGC_predicted.append(predictedVals[idx])

        all_high_CMR_p.extend(highGC_predicted)
        all_high_CMR_t.extend(highGC_observed)
        all_mid_CMR_p.extend(midGC_predicted)
        all_mid_CMR_t.extend(midGC_observed)
        all_low_CMR_p.extend(lowGC_predicted)
        all_low_CMR_t.extend(lowGC_observed)
        
        #PlotColoredGCValuesOfPredictions(currentChrom,highGC_observed,highGC_predicted,midGC_observed,midGC_predicted,lowGC_observed,lowGC_predicted)

    highGC_mean = mean(all_high_CMR_t) 
    midGC_mean = mean(all_mid_CMR_t)
    lowGC_mean = mean(all_low_CMR_t)

    high_deviations = []
    mid_deviations = []
    low_deviations = []

    rawErrors_high = []
    rawErrors_mid = []
    rawErrors_low = []

    #CALCULATE THE DEVIATIONS FROM THE TARGET PER GC_CONTENT BIN
    for idx, target in enumerate(all_high_CMR_t):
        high_deviations.append((target - highGC_mean)**2)
        rawErrors_high.append((target - all_high_CMR_p[idx])**2)

    
    for idx, target in enumerate(all_mid_CMR_t):
        mid_deviations.append((target - midGC_mean)**2)
        rawErrors_mid.append((target - all_mid_CMR_p[idx])**2)

    for idx, target in enumerate(all_low_CMR_t):
        low_deviations.append((target - lowGC_mean)**2)
        rawErrors_low.append((target - all_low_CMR_p[idx])**2)

    
    print("High GC-Content Mean: ", highGC_mean)
    print("Mid GC-content Mean: ", midGC_mean)
    print("Low GC-content Mean: ", lowGC_mean)
    Plot_GC_Deviations(all_high_CMR_t,high_deviations,all_mid_CMR_t,mid_deviations,all_low_CMR_t,low_deviations, rawErrors_low, rawErrors_mid, rawErrors_high)

def Plot_GC_Deviations(high_targets, high_devs, mid_targets, mid_devs, low_targets, low_devs, MSE_low, MSE_mid, MSE_high):

    #rho_high, rho_p_high = getSpearman(high_targets,high_devs)
    #rho_mid,rho_p_mid = getSpearman(mid_targets,mid_devs)
    #rho_low, rho_p_low =getSpearman(low_targets,low_devs)

    r_high_t, r_p_high_t = getPearson(high_devs,high_targets)
    r_mid_t, r_p_mid_t = getPearson(mid_devs,mid_targets)
    r_low_t, r_p_low_t = getPearson(low_devs,low_targets)

    r_high_d, r_p_high_d = getPearson(high_devs,MSE_high)
    r_mid_d, r_p_mid_d = getPearson(mid_devs,MSE_mid)
    r_low_d, r_p_low_d = getPearson(low_devs,MSE_low)

    #stats_box = '\n'.join((r'$\mathrm{\rho}=%.3f$' % (rVal),r'$\mathrm{p}{-}\mathrm{value}=%.2e$' % (pVal)))
    stats_pearson_h_t = ', '.join((r'$r=%.2f$' % (r_high_t),r'$p<$' + f'{r_p_high_t + 0.001:.3f}'))
    stats_pearson_m_t = ', '.join((r'$r=%.2f$' % (r_mid_t),r'$p<$'  + f'{r_p_mid_t + 0.001:.3f}'))
    stats_pearson_l_t = ', '.join((r'$r=%.2f$' % (r_low_t),r'$p<$'  + f'{r_p_low_t + 0.001:.3f}'))

    stats_pearson_h_d = ', '.join((r'$r=%.2f$' % (r_high_d),r'$p<$'+ f'{r_p_high_d + 0.001:.3f}'))
    stats_pearson_m_d = ', '.join((r'$r=%.2f$' % (r_mid_d),r'$p<$' + f'{r_p_mid_d + 0.001:.3f}'))
    stats_pearson_l_d = ', '.join((r'$r=%.2f$' % (r_low_d),r'$p<$' + f'{r_p_low_d + 0.001:.3f}'))
    #stats_pearson = ', '.join((r'Pearson: $\mathrm{\rho}=%.3f$' % (rVal_pearson),r'$\mathrm{p}{-}\mathrm{value}=%.2e$' % (pVal_pearson)))
    #stats_box = (r"$\mathrm{\rho}=%.3f \t \mathrm{p}{-}\mathrm{value}=%.2e$" % (rVal,pVal))
    props = dict(boxstyle='round', facecolor='white', alpha=0.2)

    all_joins = [ [stats_pearson_l_t,stats_pearson_l_d], [stats_pearson_m_t,stats_pearson_m_d], [stats_pearson_h_t,stats_pearson_h_d] ]
    #all_joins = [ stats_pearson_l_d, stats_pearson_m_d, stats_pearson_h_d ]


    fig, ax = plt.subplots(nrows=3, ncols=2) #layout='constrained'
    fig.tight_layout(pad = 2)
    plt.subplots_adjust(wspace = 0.1, hspace = 0.1)
    fig.set_figheight(11) #1080 pixels
    fig.set_figwidth(7) #1920 pixels
    #fig.supylabel('Predicted CMR', fontsize = 26) #default is x=0.5 and y= 0.01
    #fig.supxlabel("Deviation From Mean",fontsize = 26)
    ax[1][0].set_ylabel('Count', fontsize = 16,labelpad= 25, weight='bold')
    ax[1][1].set_ylabel('Error', fontsize = 16,labelpad= 25, weight='bold')



    #GO Through Each Subplot
    for x in range(3):
        for y in range(2):

            ax[x][y].tick_params(axis = 'x', labelsize = 8, pad=5)
            ax[x][y].set_ylim([-0.025,1])
            ax[x][y].set_xlim([-0.025,0.6])
            ax[x][y].set_yticks(np.arange(0,1.1,0.1))
            ax[x][y].set_xticks(np.arange(0,0.6,0.1))

            if y == 1:
                #ax[x][y].set_yticks(yaxis_ticks_1, labels = right_ylabels, ha='right')
                ax[x][y].tick_params(axis = 'y', labelsize = 12, pad=30)
                ax[x][y].yaxis.set_label_position("right")
                ax[x][y].yaxis.set_ticks_position('right')
                #ax[x][y].set_ylim(ylim_1)
                #ax[x][y].vlines(x=0,ymin=-1,ymax=1, color='black',linewidth=1,alpha = 1,zorder=4)
                #ax[x][y].hlines(y=0,xmin=min(xlim_right),xmax=max(xlim_right), color='black',linewidth=1,alpha = 1,zorder=4)
                ax[x][y].text(0.9, 0.91, all_joins[x][y], transform=ax[x][y].transAxes, fontsize=10, verticalalignment='bottom',horizontalalignment='right', bbox=props)
                #ax[x][y].set_xticks(xticks, labels = xticks, rotation = 45)
                #ax[x][y].set_xlim(xlim_right)
            
            else:
                ax[x][y].tick_params(axis = 'y', labelsize = 12, pad=5)
                #ax[x][y].set_yticks(left_ylabels)
                #ax[x][y].set_ylim(ylim_0)
                #ax[x][y].set_xticks(binz, labels = binz, rotation = 45)
                #ax[x][y].set_xlim(xlim_left)
                ax[x][y].text(0.9, 0.91, all_joins[x][y], transform=ax[x][y].transAxes, fontsize=10, verticalalignment='bottom',horizontalalignment='right', bbox=props)
                #ax[x][y].vlines(x=0,ymin=0,ymax=(ytick_left_max - ytick_left_step), color='black',linewidth=1,alpha = 1,zorder=4)


            ax[x][y].grid(linewidth = 0.2,zorder=1)
        
            if x != 2:
                ax[x][y].set_xticklabels([])
    
    #ax[x][y].set_ylim([-0.05,1.05])
    #ax[x][1].set_yticks(yaxis_ticks_1, labels = right_ylabels, ha='right')
    #ax[x].set_xlim([-0.01,1.025])
    #ax[x][1].set_ylim([-0.05,1.05])
    #ax[x].set_xlim([-0.01,1.025])
    #ax[x][y].set_yticks(np.arange(min(yaxis_ticks), max(yaxis_ticks), 0.1))
    #binz = [0,0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    #binz = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    
    #hp0 = ax[0][0].hist(lowGC_devs, bins=binz, alpha=1, color='#fdae61',edgecolor = 'darkgoldenrod', label =( u'$GC < 0.40$'),zorder=3)
    #hp1 = ax[1][0].hist(midGC_devs, bins=binz, alpha=1, color='#2c7bb6',edgecolor = 'mediumblue', label =( u'$ 0.45 > GC \geq 0.40$'),zorder=3)
    #hp2 = ax[2][0].hist(highGC_devs, bins=binz, alpha=1, color='#d7191c',edgecolor = 'darkred', label =( u'$GC \geq 0.45$'),zorder=3)

    #low_total_count = len(lowGC_devs)
    #high_total_count = len(highGC_devs)
    #mid_total_count = len(midGC_devs)
    #supplementary_Plot_GC_Deviations_Info(hp2,hp1,hp0, high_total_count, mid_total_count, low_total_count)

    l1_2 = ax[0][0].scatter(low_devs,low_targets, s =1, c= '#fdae61', label =( u'$GC < 0.40$'),zorder=4) 
    l1_1 = ax[1][0].scatter(mid_devs,mid_targets, s =1, c= '#2c7bb6', label =( u'$ 0.45 > GC \geq 0.40$'),zorder=3)
    l1_0 = ax[2][0].scatter(high_devs,high_targets, s =1, c= '#d7191c', label =( u'$GC \geq 0.45$'),zorder=5)

    l1_21 = ax[0][1].scatter(low_devs, MSE_low, s =1, c= '#fdae61', zorder=3, clip_on=False, alpha = 1)
    l1_11 = ax[1][1].scatter(mid_devs, MSE_mid, s =1, c= '#2c7bb6', zorder=3, clip_on=False, alpha = 1)
    l1_01 = ax[2][1].scatter(high_devs, MSE_high, s =1, c= '#d7191c', zorder=3, clip_on=False, alpha = 1)

    '''
    #column xticks:
    for x in range(3):
        shown_left_xticks = ax[x][0].xaxis.get_major_ticks()
        shown_right_xticks = ax[x][1].xaxis.get_major_ticks()
        
        #set all inviisble first
        for idx, tick in enumerate(shown_left_xticks): #slice notation: a[start_index:end_index:step]
            tick.label1.set_visible(False)
            tick.tick1line.set_visible(False)

        if ANIMAL == "Mouse":
            spacing = 4
        else:
            spacing = 8
              
        #set visible what you want:
        for idx, tick in enumerate(shown_left_xticks[::spacing]): #slice notation: a[start_index:end_index:step]
            tick.label1.set_visible(True)
            tick.tick1line.set_visible(True)
    '''



    #shown_left_xticks = ax[2][1].xaxis.get_major_ticks()
    #for tick in shown_left_xticks[::2]: #slice notation: a[start_index:end_index:step]
    #    tick.label1.set_visible(False)


    fig.text(0.5, 0.03, 'Difference of Bin GC from Mean Class GC', ha='center', va='center',fontsize=14, weight = 'bold')
    #ax.legend(loc='center',bbox_to_anchor=(0.50, 1.05), prop={'size': 20}, framealpha=0.8, ncol=4, columnspacing = 1)
    #ax.text(0.15, 1.09, stats_spearman, transform=ax.transAxes, fontsize=20, verticalalignment='bottom',horizontalalignment='left', bbox=props)
    #ax.text(0.17, 1.14, stats_pearson, transform=ax.transAxes, fontsize=20, verticalalignment='bottom',horizontalalignment='left', bbox=props)
    fig.legend(loc = 'center',prop={'size': 10},bbox_to_anchor = [0.50, 0.98],ncol=3,markerscale=5) # columnspacing = 1
    #plt.savefig( (ALLCHROMCHARTS_PATH + '_deviations_' + ANIMAL +".png" ), dpi = 300, bbox_inches='tight', pad_inches=0.1)
    #plt.clf() #close the current figure that's open
    plt.show()

def supplementary_Plot_GC_Deviations_Info(high, mid, low, high_total,mid_total, low_total):

    high_bin_counts = high[0]
    mid_bin_counts = mid[0]
    low_bin_counts = low[0]

    count = 0
    for idx,x in enumerate(high_bin_counts):
        if high[1][idx] >= 0.01:
            count += x

    print("The percentage of bins above the mean for high GC-content bins is: ",(count/(low_total + mid_total + high_total))*100)


    count = 0
    for idx,x in enumerate(low_bin_counts):
        if low[1][idx] <= -0.01:
            count += x
        else:
            break

    print("The percentage of bins below the mean for low GC-content bins is: ",(count/(low_total + mid_total + high_total)*100))

def Plot_GC_DeviationsV2(high_targets, high_devs, mid_targets, mid_devs, low_targets, low_devs, rawErrors_low, rawErrors_mid, rawErrors_high, highGC_devs, midGC_devs, lowGC_devs):

    #rho_high, rho_p_high = getSpearman(high_targets,high_devs)
    #rho_mid,rho_p_mid = getSpearman(mid_targets,mid_devs)
    #rho_low, rho_p_low =getSpearman(low_targets,low_devs)

    #r_high_t, r_p_high_t = getPearson(high_devs,high_targets)
    #r_mid_t, r_p_mid_t = getPearson(mid_devs,mid_targets)
    #r_low_t, r_p_low_t = getPearson(low_devs,low_targets)

    r_high_d, r_p_high_d = getPearson(highGC_devs,rawErrors_high)
    r_mid_d, r_p_mid_d = getPearson(midGC_devs,rawErrors_mid)
    r_low_d, r_p_low_d = getPearson(lowGC_devs,rawErrors_low)

    #stats_box = '\n'.join((r'$\mathrm{\rho}=%.3f$' % (rVal),r'$\mathrm{p}{-}\mathrm{value}=%.2e$' % (pVal)))
    #stats_pearson_h_t = ', '.join((r'$r=%.2f$' % (r_high_t),r'$p<$' + f'{r_p_high_t + 0.001:.3f}'))
    #stats_pearson_m_t = ', '.join((r'$r=%.2f$' % (r_mid_t),r'$p<$'  + f'{r_p_mid_t + 0.001:.3f}'))
    #stats_pearson_l_t = ', '.join((r'$r=%.2f$' % (r_low_t),r'$p<$'  + f'{r_p_low_t + 0.001:.3f}'))

    stats_pearson_h_d = ', '.join((r'$r=%.2f$' % (r_high_d),r'$p<$'+ f'{r_p_high_d + 0.001:.3f}'))
    stats_pearson_m_d = ', '.join((r'$r=%.2f$' % (r_mid_d),r'$p<$' + f'{r_p_mid_d + 0.001:.3f}'))
    stats_pearson_l_d = ', '.join((r'$r=%.2f$' % (r_low_d),r'$p<$' + f'{r_p_low_d + 0.001:.3f}'))
    #stats_pearson = ', '.join((r'Pearson: $\mathrm{\rho}=%.3f$' % (rVal_pearson),r'$\mathrm{p}{-}\mathrm{value}=%.2e$' % (pVal_pearson)))
    #stats_box = (r"$\mathrm{\rho}=%.3f \t \mathrm{p}{-}\mathrm{value}=%.2e$" % (rVal,pVal))
    props = dict(boxstyle='round', facecolor='white', alpha=0.2)

    #all_joins = [ [stats_pearson_l_t,stats_pearson_l_d], [stats_pearson_m_t,stats_pearson_m_d], [stats_pearson_h_t,stats_pearson_h_d] ]
    all_joins = [ stats_pearson_l_d, stats_pearson_m_d, stats_pearson_h_d ]

    #print(stats_pearson_h_d)
    #print(stats_pearson_m_d)
    #print(stats_pearson_l_d)

    fig, ax = plt.subplots(nrows=3, ncols=2) #layout='constrained'
    fig.tight_layout(pad = 2)
    plt.subplots_adjust(wspace = 0.1, hspace = 0.1)
    fig.set_figheight(11) #1080 pixels
    fig.set_figwidth(7) #1920 pixels
    #fig.supylabel('Predicted CMR', fontsize = 26) #default is x=0.5 and y= 0.01
    #fig.supxlabel("Deviation From Mean",fontsize = 26)
    ax[1][0].set_ylabel('Count', fontsize = 16,labelpad= 25, weight='bold')
    ax[1][1].set_ylabel('Error', fontsize = 16,labelpad= 25, weight='bold')

    if ANIMAL == "Mouse":

        xtick_min = -0.04
        xtick_max = 0.09
        xtick_step = 0.01

        xticks_unrounded = np.arange(xtick_min,xtick_max,xtick_step)
        xticks = [round(x,2)+0 for x in xticks_unrounded]

        xlim_right = [xtick_min,xtick_max - xtick_step]

        ytick_left_min = 0
        ytick_left_max = 280
        ytick_left_step = 20

        yaxis_ticks_1 = np.arange(-0.8, 1, 0.2)
        right_ylabels = [round(x,1)+0 for x in yaxis_ticks_1] #we add the 0 becaue we want to get rid of the signed 0 for labels
        left_ylabels = np.arange(ytick_left_min,ytick_left_max,ytick_left_step)
        ylim_0 = [ytick_left_min,ytick_left_max - ytick_left_step]
        ylim_1 = [-0.8,0.8]
        xticks_left_unrounded = np.arange(-0.04,0.09,0.0025)
        binz = [round(x,4)+0 for x in xticks_left_unrounded]
        xlim_left = [min(binz),max(binz)]
    
    else:

        xtick_min = -0.05
        xtick_max = 0.17
        xtick_step = 0.02

        xticks_unrounded = np.arange(xtick_min, xtick_max, xtick_step)
        xticks = [round(x,2)+0 for x in xticks_unrounded]
        xlim_right = [xtick_min,xtick_max - xtick_step]

        ytick_left_min = 0
        ytick_left_max = 320
        ytick_left_step = 20

        yaxis_ticks_1 = np.arange(-1.0,1.2,0.2)
        right_ylabels = [round(x,1)+0 for x in yaxis_ticks_1] #we add the 0 becaue we want to get rid of the signed 0 for labels
        left_ylabels = np.arange(ytick_left_min,ytick_left_max,ytick_left_step)
        ylim_0 = [ytick_left_min,ytick_left_max - ytick_left_step]
        ylim_1 = [-0.8,0.8]
        xticks_left_unrounded = np.arange(-0.05,0.16,0.0025)
        binz = [round(x,4)+0 for x in xticks_left_unrounded]
        xlim_left = [min(binz),max(binz)]


    #GO Through Each Subplot
    for x in range(3):
        for y in range(2):

            ax[x][y].tick_params(axis = 'x', labelsize = 8, pad=5)

            if y == 1:
                ax[x][y].set_yticks(yaxis_ticks_1, labels = right_ylabels, ha='right')
                ax[x][y].tick_params(axis = 'y', labelsize = 12, pad=30)
                ax[x][y].yaxis.set_label_position("right")
                ax[x][y].yaxis.set_ticks_position('right')
                ax[x][y].set_ylim(ylim_1)
                ax[x][y].vlines(x=0,ymin=-1,ymax=1, color='black',linewidth=1,alpha = 1,zorder=4)
                ax[x][y].hlines(y=0,xmin=min(xlim_right),xmax=max(xlim_right), color='black',linewidth=1,alpha = 1,zorder=4)
                #ax[x][y].text(0.9, 0.91, all_joins[x], transform=ax[x][y].transAxes, fontsize=10, verticalalignment='bottom',horizontalalignment='right', bbox=props)
                ax[x][y].set_xticks(xticks, labels = xticks, rotation = 45)
                ax[x][y].set_xlim(xlim_right)
            
            else:
                ax[x][y].tick_params(axis = 'y', labelsize = 12, pad=5)
                ax[x][y].set_yticks(left_ylabels)
                ax[x][y].set_ylim(ylim_0)
                ax[x][y].set_xticks(binz, labels = binz, rotation = 45)
                ax[x][y].set_xlim(xlim_left)
                ax[x][y].vlines(x=0,ymin=0,ymax=(ytick_left_max - ytick_left_step), color='black',linewidth=1,alpha = 1,zorder=4)


            ax[x][y].grid(linewidth = 0.2,zorder=1)
        
            if x != 2:
                ax[x][y].set_xticklabels([])
    
    #ax[x][y].set_ylim([-0.05,1.05])
    #ax[x][1].set_yticks(yaxis_ticks, labels = right_ylabels, ha='right')
    #ax[x].set_xlim([-0.01,1.025])
    #ax[x][1].set_ylim([-0.05,1.05])
    #ax[x].set_xlim([-0.01,1.025])
    #ax[x][y].set_yticks(np.arange(min(yaxis_ticks), max(yaxis_ticks), 0.1))
    #binz = [0,0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    #binz = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    
    hp0 = ax[0][0].hist(lowGC_devs, bins=binz, alpha=1, color='#fdae61',edgecolor = 'darkgoldenrod', label =( u'$GC < 0.40$'),zorder=3)
    hp1 = ax[1][0].hist(midGC_devs, bins=binz, alpha=1, color='#2c7bb6',edgecolor = 'mediumblue', label =( u'$ 0.45 > GC \geq 0.40$'),zorder=3)
    hp2 = ax[2][0].hist(highGC_devs, bins=binz, alpha=1, color='#d7191c',edgecolor = 'darkred', label =( u'$GC \geq 0.45$'),zorder=3)

    low_total_count = len(lowGC_devs)
    high_total_count = len(highGC_devs)
    mid_total_count = len(midGC_devs)
    supplementary_Plot_GC_Deviations_Info(hp2,hp1,hp0, high_total_count, mid_total_count, low_total_count)

    #l1_2 = ax[0][0].scatter(low_devs,low_targets, s =1, c= '#fdae61', label =( u'$GC < 0.40$'),zorder=4) 
    #l1_1 = ax[1][0].scatter(mid_devs,mid_targets, s =1, c= '#2c7bb6', label =( u'$ 0.45 > GC \geq 0.40$'),zorder=3)
    #l1_0 = ax[2][0].scatter(high_devs,high_targets, s =1, c= '#d7191c', label =( u'$GC \geq 0.45$'),zorder=5)

    l1_21 = ax[0][1].scatter(lowGC_devs, rawErrors_low, s =1, c= '#fdae61', zorder=3, clip_on=False, alpha = 0.25)
    l1_11 = ax[1][1].scatter(midGC_devs, rawErrors_mid, s =1, c= '#2c7bb6', zorder=3, clip_on=False, alpha = 0.25)
    l1_01 = ax[2][1].scatter(highGC_devs, rawErrors_high, s =1, c= '#d7191c', zorder=3, clip_on=False, alpha = 0.25)

    #column xticks:
    for x in range(3):
        shown_left_xticks = ax[x][0].xaxis.get_major_ticks()
        shown_right_xticks = ax[x][1].xaxis.get_major_ticks()
        
        #set all inviisble first
        for idx, tick in enumerate(shown_left_xticks): #slice notation: a[start_index:end_index:step]
            tick.label1.set_visible(False)
            tick.tick1line.set_visible(False)

        if ANIMAL == "Mouse":
            spacing = 4
        else:
            spacing = 8
              
        #set visible what you want:
        for idx, tick in enumerate(shown_left_xticks[::spacing]): #slice notation: a[start_index:end_index:step]
            tick.label1.set_visible(True)
            tick.tick1line.set_visible(True)


    #EXTRA BLOCK 1: START: USE THE FOLLOWING CODE TO GET BIN PERCENTAGES FOR GC_CONTENT CLASS:
    gc_class = highGC_devs
    dev_min = -1
    dev_max = 1
    mse_min = -0.3
    mse_max = 0.3
    dev_range = 0
    mse_range = 0
    for idx,dev in enumerate(gc_class):
        if dev > dev_min and dev < dev_max:
            dev_range += 1
            if rawErrors_low[idx] > mse_min and rawErrors_low[idx] < mse_max:
                mse_range += 1
    
    print("The total inside the range is: ", (mse_range / dev_range)*100 )
    #EXTRA BLOCK 1: END

    #shown_left_xticks = ax[2][1].xaxis.get_major_ticks()
    #for tick in shown_left_xticks[::2]: #slice notation: a[start_index:end_index:step]
    #    tick.label1.set_visible(False)


    fig.text(0.5, 0.03, 'Difference of Bin GC from Mean Class GC', ha='center', va='center',fontsize=14, weight = 'bold')
    #ax.legend(loc='center',bbox_to_anchor=(0.50, 1.05), prop={'size': 20}, framealpha=0.8, ncol=4, columnspacing = 1)
    #ax.text(0.15, 1.09, stats_spearman, transform=ax.transAxes, fontsize=20, verticalalignment='bottom',horizontalalignment='left', bbox=props)
    #ax.text(0.17, 1.14, stats_pearson, transform=ax.transAxes, fontsize=20, verticalalignment='bottom',horizontalalignment='left', bbox=props)
    fig.legend(loc = 'center',prop={'size': 10},bbox_to_anchor = [0.50, 0.98],ncol=3,markerscale=5) # columnspacing = 1
    plt.savefig( (ALLCHROMCHARTS_PATH + '_deviations_' + ANIMAL +".png" ), dpi = 300, bbox_inches='tight', pad_inches=0.1)
    #plt.clf() #close the current figure that's open
    #plt.show()

def Deviations_by_GC_V2():

    #returns colored predicted and observed vals, mainly used for the neural network 
    gc_df = pd.read_csv(GC_CONTENT_PATH, sep = ",", header = None, comment = '#')
    gc_df.columns = ['Chromosome','startBin', 'endBin','Aprop', 'Bprop','gcContent','atContent']
    
    #Preds
    all_low_CMR_p = []
    all_mid_CMR_p = []
    all_high_CMR_p = []

    #Targets
    all_low_CMR_t = []
    all_mid_CMR_t = []
    all_high_CMR_t = []

    #GC
    all_high_GC_content = []
    all_mid_GC_content = []
    all_low_GC_content = []

    #IN THIS SECTION GET THE MEAN CMR PER GC-CONTENT BIN:
    for chromosome in range(CHROMOSOMES):
        currentChrom = 'chr' + str(chromosome + 1)
        print('working on: ' + currentChrom)

        if (chromosome+1) < 10:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + '0' + str((chromosome + 1)) + 'TestPredictions.txt'
        else:
            chromosome_path = ALL_CHR_PREDICTIONS_PATH + str((chromosome + 1)) + 'TestPredictions.txt'

        predictions_df = pd.read_csv(chromosome_path, sep=',', header=None, comment = '#')
        predictions_df.columns = ['BinID','Predicted', 'Observed']

        gc_chr = gc_df[gc_df['Chromosome'] == currentChrom]
        gcContentList= gc_chr['gcContent'].to_numpy()
        gcTargetList = gc_chr['Aprop'].to_numpy()


        predictionBins = predictions_df['BinID'].to_numpy()
        predictedVals = predictions_df['Predicted'].to_numpy()
        observedVals = predictions_df['Observed'].to_numpy()

        highGC_observed = []
        highGC_predicted = []
        midGC_observed = []
        midGC_predicted = []
        lowGC_observed = []
        lowGC_predicted = []
        highGC_values = []
        midGC_values = []
        lowGC_values = []

        for idx,bin in enumerate(predictionBins): 

            if gcContentList[int(bin)-1] >= HIGH_GC:
                highGC_observed.append(observedVals[idx])
                highGC_predicted.append(predictedVals[idx])
                highGC_values.append(gcContentList[int(bin)-1])
            
            elif gcContentList[int(bin)-1] >= MID_GC:
                midGC_observed.append(observedVals[idx])
                midGC_predicted.append(predictedVals[idx])
                midGC_values.append(gcContentList[int(bin)-1])
            
            else:
                lowGC_observed.append(observedVals[idx])
                lowGC_predicted.append(predictedVals[idx])
                lowGC_values.append(gcContentList[int(bin)-1])

        all_high_CMR_p.extend(highGC_predicted)
        all_high_CMR_t.extend(highGC_observed)
        all_mid_CMR_p.extend(midGC_predicted)
        all_mid_CMR_t.extend(midGC_observed)
        all_low_CMR_p.extend(lowGC_predicted)
        all_low_CMR_t.extend(lowGC_observed)

        all_high_GC_content.extend(highGC_values)
        all_mid_GC_content.extend(midGC_values)
        all_low_GC_content.extend(lowGC_values)
        
        #PlotColoredGCValuesOfPredictions(currentChrom,highGC_observed,highGC_predicted,midGC_observed,midGC_predicted,lowGC_observed,lowGC_predicted)

    #lets get performance info based on gc-content:

    rho_h, rho_h_p = getSpearman(all_high_CMR_t,all_high_CMR_p)
    rho_m, rho_m_p = getSpearman(all_mid_CMR_t,all_mid_CMR_p)
    rho_l, rho_l_p = getSpearman(all_low_CMR_t,all_low_CMR_p)

    r_h, r_h_p = getPearson(all_high_CMR_t,all_high_CMR_p)
    r_m, r_m_p = getPearson(all_mid_CMR_t,all_mid_CMR_p)
    r_l, r_l_p = getPearson(all_low_CMR_t,all_low_CMR_p)

    print("Spearman high: ",rho_h, " | (",rho_h_p,")")
    print("Spearman mid: ",rho_m, " | (",rho_m_p,")")
    print("Spearman low: ",rho_l, " | (",rho_l_p,")")
    print("***********************************************")
    print("Pearson high: ",r_h, " | (",r_h_p, ")")
    print("Pearson mid: ",r_m, " | (",r_m_p, ")")
    print("Pearson low: ",r_l, " | (",r_l_p, ")") 

    low_mse = metrics.mean_squared_error(all_low_CMR_t,all_low_CMR_p)
    mid_mse = metrics.mean_squared_error(all_mid_CMR_t,all_mid_CMR_p)
    high_mse = metrics.mean_squared_error(all_high_CMR_t,all_high_CMR_p)
    print("The low GC-content MSE: ",low_mse)
    print("The mid GC-content MSE: ",mid_mse)
    print("The high GC-content MSE: ",high_mse)


    highGC_mean_CMR = mean(all_high_CMR_t) 
    midGC_mean_CMR = mean(all_mid_CMR_t)
    lowGC_mean_CMR = mean(all_low_CMR_t)

    meanHighGC_content = mean(all_high_GC_content)
    meanMidGC_content = mean(all_mid_GC_content)
    meanLowGC_content = mean(all_low_GC_content)
    meanGC_content_species = mean(all_low_GC_content + all_mid_GC_content + all_high_GC_content)

    high_deviations = []
    mid_deviations = []
    low_deviations = []

    rawErrors_high = []
    rawErrors_mid = []
    rawErrors_low = []

    deviationsInHighGC = []
    deviationsInMidGC = []
    deviationsInLowGC = []

    #CALCULATE THE DEVIATIONS FROM THE TARGET PER GC_CONTENT BIN
    for idx, target in enumerate(all_high_CMR_t):
        high_deviations.append((target - highGC_mean_CMR)**2)
        rawErrors_high.append((target - all_high_CMR_p[idx]))
    
    for idx, target in enumerate(all_mid_CMR_t):
        mid_deviations.append((target - midGC_mean_CMR)**2)
        rawErrors_mid.append((target - all_mid_CMR_p[idx]))

    for idx, target in enumerate(all_low_CMR_t):
        low_deviations.append((target - lowGC_mean_CMR)**2)
        rawErrors_low.append((target - all_low_CMR_p[idx]))

    #GC-Content:
    for idx, gc_value in enumerate(all_high_GC_content):
        deviationsInHighGC.append(gc_value - meanHighGC_content)
    
    for idx, gc_value in enumerate(all_mid_GC_content):
        deviationsInMidGC.append(gc_value - meanMidGC_content)

    for idx, gc_value in enumerate(all_low_GC_content):
        deviationsInLowGC.append(gc_value - meanLowGC_content)

    
    print("High GC-Content Mean Target CMR: ", highGC_mean_CMR)
    print("Mid GC-content Mean Target CMR: ", midGC_mean_CMR)
    print("Low GC-content Mean Target CMR: ", lowGC_mean_CMR)

    print("High GC-content mean GC: ",meanHighGC_content)
    print("Mid GC-content mean GC: ",meanMidGC_content)
    print("Low GC-content mean GC: ",meanLowGC_content)

    Plot_GC_DeviationsV2(all_high_CMR_t,high_deviations,all_mid_CMR_t,mid_deviations,all_low_CMR_t,low_deviations, rawErrors_low, rawErrors_mid, rawErrors_high, deviationsInHighGC, deviationsInMidGC, deviationsInLowGC)

    #check over-estimated vs under-estimated for the raw errors:
    #LOW
    errorAbove0_tally = 0
    errorBelow0_tally = 0
    for x in rawErrors_low:
        if x > 0:
            errorAbove0_tally += 1
        elif x < 0:
            errorBelow0_tally += 1
        else:
            pass
    
    print("Errors above 0 in Low GC: ", (errorAbove0_tally / (errorAbove0_tally + errorBelow0_tally))*100)

    #MID
    errorAbove0_tally = 0
    errorBelow0_tally = 0
    for x in rawErrors_mid:
        if x > 0:
            errorAbove0_tally += 1
        elif x < 0:
            errorBelow0_tally += 1
        else:
            pass
    
    print("Errors above 0 in mid GC: ", (errorAbove0_tally / (errorAbove0_tally + errorBelow0_tally))*100)

    #HIGH
    errorAbove0_tally = 0
    errorBelow0_tally = 0
    for x in rawErrors_high:
        if x > 0:
            errorAbove0_tally += 1
        elif x < 0:
            errorBelow0_tally += 1
        else:
            pass
    
    print("Errors above 0 in high GC: ", (errorAbove0_tally / (errorAbove0_tally + errorBelow0_tally))*100)


    """
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.tight_layout(pad = 2)
    plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    fig.set_figheight(11.25) #1080 pixels
    fig.set_figwidth(20) #1920 pixels
    xaxis_ticks = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]
    yaxis_ticks = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]
    
    #x[0].set_xlabel('A-Proportion', fontsize = 26,labelpad= 25)
    ax.set_aspect(1)
    ax.set_ylabel('Predicted CMR', fontsize = 26,labelpad= 25, weight ='bold')
    ax.set_xlabel('Target CMR', fontsize = 26,labelpad= 25, weight = 'bold')
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis = 'x', labelsize = 22, pad=10)
    ax.tick_params(axis = 'y', labelsize = 22, pad=10)


    ax.set_ylim([0.0,1.0])
    ax.set_xlim([0,1])
    #ax.set_yticks(np.arange(min(yaxis_ticks), max(yaxis_ticks), 0.1))
    #ax.set_xticks(np.arange(min(xaxis_ticks), max(xaxis_ticks), 0.1))

    l1_0 = ax.scatter(all_mid_CMR_t,all_mid_CMR_p, s =2, c= '#d7191c', label =( u'$GC \geq 0.45$'),zorder=5)
    l2 = ax.plot([0.1,0.9],[0.1,0.9], color='black', linestyle = '-',zorder=4, linewidth=3)

    plt.show()
    """

def Predictions_T_Test():
    
    print("MOUSE CHROMOSOMES: ")
    #mouse:
    gc_df = pd.read_csv('./data/mouse/mouse_gc_content.csv', sep = ",", header = None, comment = '#')
    gc_df.columns = ['Chromosome','startBin', 'endBin','Aprop', 'Bprop','gcContent','atContent']

    m_lowGC_CMRpreds = []
    m_midGC_CMRpreds = []
    m_highGC_CMRpreds = []

    for chromosome in range(19):
        currentChrom = 'chr' + str(chromosome + 1)
        print('working on Mouse: ' + currentChrom)

        if (chromosome+1) < 10:
            chromosome_path = './data/mouse/'+ job +'/all_predictions/' + '0' + str((chromosome + 1)) + 'TestPredictions.txt'
        else:
            chromosome_path = './data/mouse/'+ job +'/all_predictions/' + str((chromosome + 1)) + 'TestPredictions.txt'

        predictions_df = pd.read_csv(chromosome_path, sep=',', header=None, comment = '#')
        predictions_df.columns = ['BinID','Predicted', 'Observed']

        gc_chr = gc_df[gc_df['Chromosome'] == currentChrom]
        gcContentList= gc_chr['gcContent'].to_numpy()
        gcTargetList = gc_chr['Aprop'].to_numpy()


        predictionBins = predictions_df['BinID'].to_numpy()
        predictedVals = predictions_df['Predicted'].to_numpy()
        observedVals = predictions_df['Observed'].to_numpy()

        for idx,bin in enumerate(predictionBins): 
            '''
            print('********************************')
            print('The predicted value from prediction file is: ',predictedVals[idx])
            print('The observed value from prediction file is: ',observedVals[idx])
            print('The GC-content for bin: ',int(bin)," is ",gcContentList[int(bin)-1])
            print('The target from the gc-content list is: ',gcTargetList[int(bin)-1])
            print('********************************')
            '''
            if gcContentList[int(bin)-1] >= HIGH_GC:
                #m_highGC_CMRtargets.append(observedVals[idx])
                m_highGC_CMRpreds.append(predictedVals[idx])
            
            elif gcContentList[int(bin)-1] >= MID_GC:
                #m_midGC_CMRtargets.append(observedVals[idx])\
                m_midGC_CMRpreds.append(predictedVals[idx])
            
            else:
                #m_lowGC_CMRtargets.append(observedVals[idx])
                m_lowGC_CMRpreds.append(predictedVals[idx])

    #put human here: 
    print("HUMAN CHROMOSOMES")
    gc_df = pd.read_csv('./data/human/human_gc_content.csv', sep = ",", header = None, comment = '#')
    gc_df.columns = ['Chromosome','startBin', 'endBin','Aprop', 'Bprop','gcContent','atContent']

    h_lowGC_CMRpreds = []
    h_midGC_CMRpreds = []
    h_highGC_CMRpreds = []

    for chromosome in range(22):
        currentChrom = 'chr' + str(chromosome + 1)
        print('working on Human: ' + currentChrom)

        if (chromosome+1) < 10:
            chromosome_path = './data/human/'+ job +'/all_predictions/' + '0' + str((chromosome + 1)) + 'TestPredictions.txt'
        else:
            chromosome_path = './data/human/'+ job +'/all_predictions/' + str((chromosome + 1)) + 'TestPredictions.txt'

        predictions_df = pd.read_csv(chromosome_path, sep=',', header=None, comment = '#')
        predictions_df.columns = ['BinID','Predicted', 'Observed']

        gc_chr = gc_df[gc_df['Chromosome'] == currentChrom]
        gcContentList= gc_chr['gcContent'].to_numpy()
        gcTargetList = gc_chr['Aprop'].to_numpy()


        predictionBins = predictions_df['BinID'].to_numpy()
        predictedVals = predictions_df['Predicted'].to_numpy()
        observedVals = predictions_df['Observed'].to_numpy()

        for idx,bin in enumerate(predictionBins): 
            '''
            print('********************************')
            print('The predicted value from prediction file is: ',predictedVals[idx])
            print('The observed value from prediction file is: ',observedVals[idx])
            print('The GC-content for bin: ',int(bin)," is ",gcContentList[int(bin)-1])
            print('The target from the gc-content list is: ',gcTargetList[int(bin)-1])
            print('********************************')
            '''
            if gcContentList[int(bin)-1] >= HIGH_GC:
                #h_highGC_CMRtargets.append(observedVals[idx])
                h_highGC_CMRpreds.append(predictedVals[idx])
            
            elif gcContentList[int(bin)-1] >= MID_GC:
                #h_midGC_CMRtargets.append(observedVals[idx])
                h_midGC_CMRpreds.append(predictedVals[idx])
            
            else:                
                #h_lowGC_CMRtargets.append(observedVals[idx])
                h_lowGC_CMRpreds.append(predictedVals[idx])

    #HUMAN GENERAL STATS:
    print('=================================================================================================================')
    print("\nHUMAN GENERAL PREDICTION STATS:\n")
    h_Median = statistics.median(h_highGC_CMRpreds)
    h_mean = mean(h_highGC_CMRpreds)
    h_std = statistics.pstdev(h_highGC_CMRpreds)
    m_Median = statistics.median(h_midGC_CMRpreds)
    m_mean = mean(h_midGC_CMRpreds)
    m_std = statistics.pstdev(h_midGC_CMRpreds)
    l_Median = statistics.median(h_lowGC_CMRpreds)
    l_mean = mean(h_lowGC_CMRpreds)
    l_std = statistics.pstdev(h_lowGC_CMRpreds)

    q1_l = np.percentile(h_lowGC_CMRpreds,25)
    q3_l = np.percentile(h_lowGC_CMRpreds,75)
    q1_m = np.percentile(h_midGC_CMRpreds,25)
    q3_m = np.percentile(h_midGC_CMRpreds,75)
    q1_h = np.percentile(h_highGC_CMRpreds,25)
    q3_h = np.percentile(h_highGC_CMRpreds,75)

    print('**************************************')
    print("MEAN HUMAN HIGH-GC PREDICTIONS: ",h_mean)
    print("MEAN HUMAN MID-GC PREDICTIONS: ", m_mean)
    print("MEAN HUMAN LOW-GC PREDICTIONS: ", l_mean)
    print("MEDIAN HUMAN HIGH-GC PREDICTIONS: ", h_Median)
    print("MEDIAN HUMAN MID-GC PREDICTIONS: ", m_Median)
    print("MEDIAN HUMAN LOW-GC PREDICTIONS: ", l_Median)

    print('STD HUMAN HIGH-GC PREDICTIONS: ',h_std)
    print('STD HUMAN MID-GC PREDICTIONS: ', m_std)
    print('STD HUMAN LOW-GC PREDICTIONS: ', l_std)
    print('**************************************')
    print("The total count of HUMAN high-GC PREDICTIONS: ",len(h_highGC_CMRpreds), " %: ", len(h_highGC_CMRpreds) / (len(h_highGC_CMRpreds) + len(h_midGC_CMRpreds) + len(h_lowGC_CMRpreds)))
    print("The total count of HUMAN mid-GC PREDICTIONS: ",len(h_midGC_CMRpreds), " %: ", len(h_midGC_CMRpreds) / (len(h_highGC_CMRpreds) + len(h_midGC_CMRpreds) + len(h_lowGC_CMRpreds)))
    print("The total count of HUMAN low-GC PREDICTIONS: ",len(h_lowGC_CMRpreds), " %: ", len(h_lowGC_CMRpreds) / (len(h_highGC_CMRpreds) + len(h_midGC_CMRpreds) + len(h_lowGC_CMRpreds)))

    # create 99% confidence interval 
    h_lowGC_CMRpreds_ci = scipy.stats.t.interval(alpha=0.99, df=len(h_lowGC_CMRpreds)-1, loc=np.mean(h_lowGC_CMRpreds),  scale=scipy.stats.sem(h_lowGC_CMRpreds))
    h_midGC_CMRpreds_ci = scipy.stats.t.interval(alpha=0.99, df=len(h_midGC_CMRpreds)-1, loc=np.mean(h_midGC_CMRpreds),  scale=scipy.stats.sem(h_midGC_CMRpreds)) 
    h_highGC_CMRpreds_ci = scipy.stats.t.interval(alpha=0.99, df=len(h_highGC_CMRpreds)-1, loc=np.mean(h_highGC_CMRpreds),  scale=scipy.stats.sem(h_highGC_CMRpreds)) 

    print('The HUMAN low GC PREDICTIONS mean CI: ',h_lowGC_CMRpreds_ci)
    print("The HUMAN mid GC PREDICTIONS mean CI: ",h_midGC_CMRpreds_ci)
    print('The HUMAN high GC PREDICTIONS mean CI: ',h_highGC_CMRpreds_ci)

    print('****************************')
    print("The Q1 HUMAN LOW-GC PREDICTIONS: ", q1_l)
    print("The Q3 HUMAN LOW-GC PREDICTIONS: ",q3_l)
    print("The Q1 HUMAN MID-GC PREDICTIONS: ", q1_m)
    print("The Q3 HUMAN MID-GC PREDICTIONS: ",q3_m)
    print("The Q1 HUMAN HIGH-GC PREDICTIONS ", q1_h)
    print("The Q3 HUMAN HIGH-GC PREDICTIONS: ",q3_h)


    #MOUSE GENERAL STATS:
    print('=================================================================================================================')
    print("\nMOUSE GENERAL PREDICTION STATS:\n")
    h_Median = statistics.median(m_highGC_CMRpreds)
    h_mean = mean(m_highGC_CMRpreds)
    h_std = statistics.pstdev(m_highGC_CMRpreds)
    m_Median = statistics.median(m_midGC_CMRpreds)
    m_mean = mean(m_midGC_CMRpreds)
    m_std = statistics.pstdev(m_midGC_CMRpreds)
    l_Median = statistics.median(m_lowGC_CMRpreds)
    l_mean = mean(m_lowGC_CMRpreds)
    l_std = statistics.pstdev(m_lowGC_CMRpreds)

    q1_l = np.percentile(m_lowGC_CMRpreds,25)
    q3_l = np.percentile(m_lowGC_CMRpreds,75)
    q1_m = np.percentile(m_midGC_CMRpreds,25)
    q3_m = np.percentile(m_midGC_CMRpreds,75)
    q1_h = np.percentile(m_highGC_CMRpreds,25)
    q3_h = np.percentile(m_highGC_CMRpreds,75)

    print('**************************************')
    print("MEAN MOUSE HIGH-GC PREDICTIONS: ",h_mean)
    print("MEAN MOUSE MID-GC PREDICTIONS: ", m_mean)
    print("MEAN MOUSE LOW-GC PREDICTIONS: ", l_mean)
    print("MEDIAN MOUSE HIGH-GC PREDICTIONS: ", h_Median)
    print("MEDIAN MOUSE MID-GC PREDICTIONS: ", m_Median)
    print("MEDIAN MOUSE LOW-GC PREDICTIONS: ", l_Median)

    print('STD MOUSE HIGH-GC PREDICTIONS: ',h_std)
    print('STD MOUSE MID-GC PREDICTIONS: ', m_std)
    print('STD MOUSE LOW-GC PREDICTIONS: ', l_std)
    print('**************************************')
    print("The total count of MOUSE high-GC PREDICTIONS: ",len(m_highGC_CMRpreds), " %: ", len(m_highGC_CMRpreds) / (len(m_highGC_CMRpreds) + len(m_midGC_CMRpreds) + len(m_lowGC_CMRpreds)))
    print("The total count of MOUSE mid-GC PREDICTIONS: ",len(m_midGC_CMRpreds), " %: ", len(m_midGC_CMRpreds) / (len(m_highGC_CMRpreds) + len(m_midGC_CMRpreds) + len(m_lowGC_CMRpreds)))
    print("The total count of MOUSE low-GC PREDICTIONS: ",len(m_lowGC_CMRpreds), " %: ", len(m_lowGC_CMRpreds) / (len(m_highGC_CMRpreds) + len(m_midGC_CMRpreds) + len(m_lowGC_CMRpreds)))

    # create 99% confidence interval 
    m_lowGC_CMRpreds_ci = scipy.stats.t.interval(alpha=0.99, df=len(m_lowGC_CMRpreds)-1, loc=np.mean(m_lowGC_CMRpreds),  scale=scipy.stats.sem(m_lowGC_CMRpreds))
    m_midGC_CMRpreds_ci = scipy.stats.t.interval(alpha=0.99, df=len(m_midGC_CMRpreds)-1, loc=np.mean(m_midGC_CMRpreds),  scale=scipy.stats.sem(m_midGC_CMRpreds)) 
    m_highGC_CMRpreds_ci = scipy.stats.t.interval(alpha=0.99, df=len(m_highGC_CMRpreds)-1, loc=np.mean(m_highGC_CMRpreds),  scale=scipy.stats.sem(m_highGC_CMRpreds)) 

    print('The MOUSE low GC PREDICTIONS mean CI: ',m_lowGC_CMRpreds_ci)
    print("The MOUSE mid GC PREDICTIONS mean CI: ",m_midGC_CMRpreds_ci)
    print('The MOUSE high GC PREDICTIONS mean CI: ',m_highGC_CMRpreds_ci)

    print('****************************')
    print("The Q1 MOUSE LOW-GC PREDICTIONS: ", q1_l)
    print("The Q3 MOUSE LOW-GC PREDICTIONS: ",q3_l)
    print("The Q1 MOUSE MID-GC PREDICTIONS: ", q1_m)
    print("The Q3 MOUSE MID-GC PREDICTIONS: ",q3_m)
    print("The Q1 MOUSE HIGH-GC PREDICTIONS ", q1_h)
    print("The Q3 MOUSE HIGH-GC PREDICTIONS: ",q3_h)
    print('=================================================================================================================')
    
    
    #T_TEST ANALYSIS:
    print("\nT-TEST ANAYLISIS:\n")
    print("Mouse Variance low: ",np.var(m_lowGC_CMRpreds))
    print("Mouse Variance mid: ",np.var(m_midGC_CMRpreds))
    print("Mouse Variance high: ",np.var(m_highGC_CMRpreds))

    m_low_stat = stats.ttest_ind(a=m_lowGC_CMRpreds, b=m_midGC_CMRpreds, equal_var=False)
    m_mid_stat = stats.ttest_ind(a=m_midGC_CMRpreds, b=m_highGC_CMRpreds, equal_var=False)
    m_high_stat = stats.ttest_ind(a=m_lowGC_CMRpreds, b=m_highGC_CMRpreds, equal_var=False)

    print("Mouse low-mid Statistic",m_low_stat)
    print("Mouse mid-high Statistic",m_mid_stat)
    print("Mouse low-high-Statistic",m_high_stat)
    print('*****************************************************************')

    print("Human Variance low: ",np.var(h_lowGC_CMRpreds))
    print("Human Variance mid: ",np.var(h_midGC_CMRpreds))
    print("Human Variance high: ",np.var(h_highGC_CMRpreds))

    h_low_stat = stats.ttest_ind(a=h_lowGC_CMRpreds, b=h_midGC_CMRpreds, equal_var=False)
    h_mid_stat = stats.ttest_ind(a=h_midGC_CMRpreds, b=h_highGC_CMRpreds, equal_var=False)
    h_high_stat = stats.ttest_ind(a=h_lowGC_CMRpreds, b=h_highGC_CMRpreds, equal_var=False)

    print("Human low-mid Statistic",h_low_stat)
    print("Human mid-high Statistic",h_mid_stat)
    print("Human low-high Statistic",h_high_stat)
    print('******************************************************************')

    mh_lowlow_compare_stat = stats.ttest_ind(a=m_lowGC_CMRpreds, b=h_lowGC_CMRpreds, equal_var=False)
    mh_lowmid_compare_stat = stats.ttest_ind(a=m_lowGC_CMRpreds, b=h_midGC_CMRpreds, equal_var=False)
    mh_lowhigh_compare_stat = stats.ttest_ind(a=m_lowGC_CMRpreds, b=h_highGC_CMRpreds, equal_var=False)


    mh_midlow_compare_stat = stats.ttest_ind(a=m_midGC_CMRpreds, b=h_lowGC_CMRpreds, equal_var=False)
    mh_midmid_compare_stat = stats.ttest_ind(a=m_midGC_CMRpreds, b=h_midGC_CMRpreds, equal_var=False)
    mh_midhigh_compare_stat = stats.ttest_ind(a=m_midGC_CMRpreds, b=h_highGC_CMRpreds, equal_var=False)


    mh_highlow_compare_stat = stats.ttest_ind(a=m_highGC_CMRpreds, b=h_lowGC_CMRpreds, equal_var=False)
    mh_highmid_compare_stat = stats.ttest_ind(a=m_highGC_CMRpreds, b=h_midGC_CMRpreds, equal_var=False)
    mh_highhigh_compare_stat = stats.ttest_ind(a=m_highGC_CMRpreds, b=h_highGC_CMRpreds, equal_var=False)

    print("Mouse-Human low-low compare: ",mh_lowlow_compare_stat)
    print("Mouse-Human low-mid compare: ",mh_lowmid_compare_stat)
    print("Mouse-Human low-high compare: ",mh_lowhigh_compare_stat)

    print("Mouse-Human mid-low compare: ",mh_midlow_compare_stat)
    print("Mouse-Human mid-mid compare: ",mh_midmid_compare_stat)
    print("Mouse-Human mid-high compare: ",mh_midhigh_compare_stat)

    print("Mouse-Human high-low compare: ",mh_highlow_compare_stat)
    print("Mouse-Human high-mid compare: ",mh_highmid_compare_stat)
    print("Mouse-Human high-high compare: ",mh_highhigh_compare_stat)
    print('=================================================================================================================')
    #print("RANDOM TEST THINGY MIAJIG: ", stats.ttest_ind(a=h_lowGC_CMRpreds, b=h_lowGC_CMRpreds, equal_var=False))


    #m_heatmap_input = [[0,m_low_stat[0],m_high_stat[0]], [m_low_stat[0],0,m_mid_stat[0]], [m_high_stat[0],m_mid_stat[0],0]]
    #h_heatmap_input = [[0,h_low_stat[0],h_high_stat[0]], [h_low_stat[0],0,h_mid_stat[0]], [h_high_stat[0],h_mid_stat[0],0]]
    heatmap_list2 = [[0,      m_low_stat[0],m_high_stat[0],mh_lowlow_compare_stat[0],mh_lowmid_compare_stat[0],mh_lowhigh_compare_stat[0]], 
               [m_low_stat[0],0,      m_mid_stat[0],mh_midlow_compare_stat[0],mh_midmid_compare_stat[0],mh_midhigh_compare_stat[0]], 
               [m_high_stat[0],m_mid_stat[0],0,       mh_highlow_compare_stat[0],mh_highmid_compare_stat[0],mh_highhigh_compare_stat[0]],    
               [mh_lowlow_compare_stat[0],mh_midlow_compare_stat[0],mh_highlow_compare_stat[0],0,      h_low_stat[0],h_high_stat[0]], 
               [mh_lowmid_compare_stat[0],mh_midmid_compare_stat[0],mh_highmid_compare_stat[0],h_low_stat[0],0,      h_mid_stat[0]], 
               [mh_lowhigh_compare_stat[0],mh_midhigh_compare_stat[0], mh_highhigh_compare_stat[0],h_high_stat[0],h_mid_stat[0],0]]
    #hm_compare = [hm_low_compare_stat[0],hm_mid_compare_stat[0],hm_high_compare_stat[0]]
    #m_heatmap_input = [m_low_stat[0],m_mid_stat[0],m_high_stat[0]]
    #h_heatmap_input = [h_low_stat[0],h_mid_stat[0],h_high_stat[0]]

    heatmap_list = []
    for x in heatmap_list2:
        temp_list = []
        for y in x:
            temp_list.append(abs(y))
        heatmap_list.append(temp_list)

    the_labels = ["Mouse Low-GC","Mouse Mid-GC","Mouse High-GC","Human Low-GC", "Human Mid-GC", "Human High-GC"]
    heatmap_array = np.array(heatmap_list)
    lower_mask = np.triu(heatmap_array)
    #

    ax = sns.heatmap(heatmap_array,linewidths=0.5,annot=True, yticklabels=the_labels,annot_kws={"size": 20},mask=lower_mask,fmt='.0f')
    ax.set_xticklabels(labels=the_labels, weight='bold',fontsize = 20)
    ax.set_yticklabels(labels=the_labels, weight='bold',fontsize = 20)
    plt.show()

        
#=======================================================================
#============================= MAIN ====================================
if __name__ == "__main__":
    
    print('Hello! You are using the CMR Analysis script, have fun!')

    #========GLOBAL PATHS=======# 
    
    HIGH_GC = 0.45
    MID_GC = 0.40
    ALL_OUTPUT = './output/hm_output/'
    #seeds = [0,1,2,3,4]
    jobs = [0]


    for x in jobs:
        
        job = str(x)

        print('######################################### JOB: ' + job)

        if(sys.argv[1] == "mouse"):
            ANIMAL = "Mouse"
            CHROMOSOMES = 19
            SINGLE_CHR_PRED_PATH = './data/mouse/PredictionResultsOnChr1/'
            GC_CONTENT_PATH = './data/mouse/mouse_gc_content.csv'
            PATH_TO_RENAMING = './data/mouse/'+ job +'/all_trainingloss/'
            ALL_CHR_PREDICTIONS_PATH = './data/mouse/'+ job +'/all_predictions/'
            TRAININGLOSS_PATH = './data/mouse/'+ job +'/all_trainingloss/'
            TRAININGLOG_PATH = './data/mouse/'+ job +'/all_traininglogs/'
            SCATTERPLOTS_SAVE_PATH = './output/mouse/scatterplots/'
            BOXPLOTS_SAVE_PATH = './output/mouse/boxplots/'
            GCPLOT_SAVE_PATH = './output/mouse/gcerrorplots/'
            LOSS_ANALYSIS_PATH = './output/mouse/trainingloss_analysis/'
            BESTWORST_LOSS_PATH = './output/mouse/trainingloss_analysis/bestworst/'
            ALLCHROMCHARTS_PATH = './output/mouse/allChromCharts/'
            LINREG_DATA_PATH = "C:/Users/rollo_tomasi/My Drive/School/UoGuelph/Masters/python_work/large4DNanalysis/apps/Lin_Log_Model/output/mouse/LinReg/mouse_linReg_Model_data.csv"
            OVERUNDER_PATH = './output/mouse/overunder/'

        
        elif(sys.argv[1] == "human"):
            ANIMAL = "Human"
            CHROMOSOMES = 22
            SINGLE_CHR_PRED_PATH = './data/human/PredictionResultsOnChr1/00TestPredictions_sigmoid.txt'
            GC_CONTENT_PATH = './data/human/human_gc_content.csv'
            ALL_CHR_PREDICTIONS_PATH = './data/human/'+ job +'/all_predictions/'
            TRAININGLOG_PATH = './data/human/'+ job +'/all_traininglogs/'
            TRAININGLOSS_PATH = './data/human/'+ job +'/all_trainingloss/'
            PATH_TO_RENAMING = './data/human/'+ job +'/all_traininglogs/'
            SCATTERPLOTS_SAVE_PATH = './output/human/scatterplots/'
            BOXPLOTS_SAVE_PATH = './output/human/boxplots/'
            GCPLOT_SAVE_PATH = './output/human/gcerrorplots/'
            LOSS_ANALYSIS_PATH = './output/human/trainingloss_analysis/'
            BESTWORST_LOSS_PATH = './output/human/trainingloss_analysis/bestworst/'
            ALLCHROMCHARTS_PATH = './output/human/allChromCharts/'
            LINREG_DATA_PATH = "C:/Users/rollo_tomasi/My Drive/School/UoGuelph/Masters/python_work/large4DNanalysis/apps/Lin_Log_Model/output/human/LinReg/human_linReg_Model_data.csv"
            OVERUNDER_PATH = './output/human/overunder/'
        
        else:
            raise Exception("Sorry, please choose 'human' or 'mouse'")


        #MAIN METHODS -- uncomment whichever method you want to use:

        #TrainingLossAnalysis(22)
        #BestWorstTrainingLoss2Charts()
        #compareGC_w_Error()
        #manyBoxPlots()
        #predictionsBoxPlot(1)
        #predictionsScatterPlot(1)
        #ManyPredictionScatterPlots(ANIMAL)
        #TestingLossAnalysis()
        #TargetCMR_GC_Distribution(19)
        #CMR_Stats()
        #Predictions_GC_Correlation_Analysis()
        #Predictions_OverOrUnderEstimated()
        #RandomPredictions()
        #RandomPredictions_AllChroms()
        #TrainingLossAnalysis()
        #TrainingLossAnalysis_WithBars()
        #TrainingLossAnalysis_AllChrom()
        #Plot_GC_VS_CMR()
        #PlotLinRegVsABCRNet()
        #Deviations_by_GC()
        #Deviations_by_GC_V2()
        #AnalyzeGCviaBoxPlot()
        #CMR_GC_Histograms()
        #Predictions_T_Test()
        #ColorGCValuesOfPredictions()
        #Target_GC_Correlation_Analysis()


        #EXTRA USEFUL METHODS: -- uncomment whatever method you want to use:
        #renameTrainingOutputFiles()
        #everyBin_Analysis()
        #everyBin_AnalysisRAW()
        #examineChromFiles()
        #NoNModelPredictionOnNChrom(1,0.0314,"Model Trained on No N's Predicting Chromosome with Ns -- ") #the loss was 0.0314
        #buildNoNComparisonDataset()
        #NModelPredictionOnNoNChrom(1,0.0293,"Model Trained on N's Predicting Chromosome without Ns -- ") #the loss is
        #nModelPerformanceComparison(1,'N-Model vs No-N-Model: Prediction Performance -- ')
        #CombineNModelDatasets(1)
    
