import os
import pandas as pd
from matplotlib import pyplot as plt 
import sklearn.metrics as metrics
import gc
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import sys
from statistics import mean
from scipy import stats
from sklearn import metrics
import statistics

def convertChannel(example):
    channel = []
    for i in range (len(example)):
        channel.append(int(example[i])) #list(map( int, example[i]))) t

    return channel

def CreatePredVsTargetChart(chrom, targetData, predData,gc_content):

    low_preds = []
    low_obs = []
    mid_preds = []
    mid_obs = []
    high_preds = []
    high_obs =[]

    #remember targetData here is basically y_test from before, same index as x_test
    for idx, target in enumerate(targetData):

        if gc_content[idx] =='high':
            high_obs.append(target)
            high_preds.append(predData[idx])
        elif gc_content[idx] == 'mid':
            mid_obs.append(target)
            mid_preds.append(predData[idx])
        else:
            low_obs.append(target)
            low_preds.append(predData[idx])


    print('... ... ...creating the prediction target chart for chr' + str(chrom))
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='col')
    fig.tight_layout(pad = 2)
    plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    fig.set_figheight(11.25) #1080 pixels
    fig.set_figwidth(20) #1920 pixels
    axis_ticks = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]

    ax.set_xlabel('Targets', fontsize = 26,labelpad= 25)
    ax.set_ylabel('Predictions', fontsize = 26,labelpad= 10.0)
    ax.set_aspect(1)
    #ax[0].spines['top'].set_visible(False)
    #ax[0].spines['right'].set_visible(False)
    #ax[0].spines['bottom'].set_visible(False)
    ax.tick_params(axis = 'x', labelsize = 22, pad=10)
    ax.tick_params(axis = 'y', labelsize = 22, pad=10)
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    ax.set_yticks(np.arange(min(axis_ticks), max(axis_ticks), 0.1))
    ax.set_xticks(np.arange(min(axis_ticks), max(axis_ticks), 0.1))
    l1_0 = ax.scatter(high_obs,high_preds, s =50,marker='d', c= 'r', label =( u'$GC \geq 0.45$'),zorder=3)
    l1_1 = ax.scatter(mid_obs,mid_preds, s =50, c= 'b', label =( u'$ 0.45 > GC \geq 0.40$'),zorder=1)
    l1_2 = ax.scatter(low_obs,low_preds, s =50,marker='s', c= 'darkorange', label =( u'$GC < 0.40$'),zorder=2)
    l2 = ax.plot([0.1,0.9],[0.1,0.9], color='black', linestyle = '-',zorder=4, linewidth=3, label= 'Perfect Prediction')
    #ax[0].axhline(y=0.5, color = 'black', linewidth = 0.5)
    ax.grid(linewidth = 0.2)
    ax.margins(0)
    ax.legend(loc='center',bbox_to_anchor=(0.50, 1.05), prop={'size': 20}, framealpha=0.8, ncol=4, columnspacing = 1)
    plt.savefig( (CHART_OUTPUT_PATH +'Pred_Target_LinModel_chr_wGC'+ str(chrom) + ".png" ), dpi = 300, bbox_inches='tight', pad_inches=0.1)
    plt.clf() #close the current figure that's open
    #plt.show()

def CreateRegressionPerformanceChart(chrom,x_data, y_data,bestFitLine, xName,yName, chartName):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_data, bestFitLine)
    mean_absolute_error=metrics.mean_absolute_error(y_data, bestFitLine)
    mse=metrics.mean_squared_error(y_data, bestFitLine)
    #mean_squared_log_error=metrics.mean_squared_log_error(y_data, bestFitLine)
    median_absolute_error=metrics.median_absolute_error(y_data, bestFitLine)
    r2=metrics.r2_score(y_data, bestFitLine)

    #NOTE: THE GC CONTENT IS MAINLY BETWEEN 0.3 and 0.6 so theres no point in having points stretching from 0.1 to 0.9
    print('... ... ...creating the ' + chartName + ' performance chart for chr' + str(chrom))
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='col')
    fig.tight_layout(pad = 2)
    plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    fig.set_figheight(11.25) #1080 pixels
    fig.set_figwidth(20) #1920 pixels
    xaxis_ticks = [0.3,0.4,0.5,0.6,0.7]
    yaxis_ticks = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]
    
    ax.set_xlabel(xName, fontsize = 26,labelpad= 25)
    ax.set_ylabel(yName, fontsize = 26,labelpad= 12.0)
    #ax[0].spines['top'].set_visible(False)
    #ax[0].spines['right'].set_visible(False)
    #ax[0].spines['bottom'].set_visible(False)
    ax.tick_params(axis = 'x', labelsize = 22, length = 0, pad=10)
    ax.tick_params(axis = 'y', labelsize = 22)
    ax.set_title("Regression Performance on Preportion of A Compartments in "+ ANIMAL + '_chr' + str(chrom) +" Given GC Content", y= 1, loc = 'center', fontsize = 28,pad=20)
    ax.set_ylim([0,1])
    ax.set_xlim([0.3,0.7])
    ax.set_yticks(np.arange(min(yaxis_ticks), max(yaxis_ticks), 0.1))
    ax.set_xticks(np.arange(min(xaxis_ticks), max(xaxis_ticks), 0.05))
    l1 = ax.scatter(x_data, y_data, s =15, c= 'b', label = 'performance')
    ax.plot(x_data,bestFitLine,color = 'r', label = 'line of best fit')

    stats = '\n'.join((r'$y=%.3fx %.3f$' % (m_val,b_val, ),
            r'$\mathrm{R^2}=%.3f$' % (r2, ), 
            r'$\mathrm{MSE}=%.3f$' % (mse, ),)) #Join all items in a tuple into a string, using a '\n' character as separator:
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.95, 0.05, stats, transform=ax.transAxes, fontsize=28, verticalalignment='bottom',horizontalalignment='right', bbox=props)

    
    #ax[0].axhline(y=0.5, color = 'black', linewidth = 0.5)
    ax.grid(linewidth = 0.2)
    ax.margins(0)
    ax.legend(prop={'size': 14})
    plt.savefig( (CHART_OUTPUT_PATH + chartName +'_LinModel_chr'+ str(chrom) + ".png" ), dpi = 100)
    plt.clf() #close the current figure that's open
    #plt.show()

def ConfusionMatrix(y_observed, y_predicted):

    confusionMat = metrics.confusion_matrix(y_observed,y_predicted, labels=[0,1])
    print('... Creating confusion matrix: ', confusionMat)
    fig,ax = plt.subplots()
    fig.set_figheight(11.25) #1080 pixels
    fig.set_figwidth(20) #1920 pixels

    sns.set(font_scale = 1.6)
    sns.heatmap(confusionMat,cmap ='Greens', annot=True,cbar_kws={'orientation':'vertical', 'label':'Count'}, xticklabels=['Not-A','A'],yticklabels=['Not-A','A'], fmt = 'd')
    ax.set_xlabel('Pedicted')
    ax.set_ylabel('Observed')
    plt.title('Linear Regression Confusion Matrix, Threshold--' + THRESHOLD_PERCENTAGE + ' --' + ANIMAL)
    plt.savefig( (CHART_OUTPUT_PATH + 'LinReg_Matrix--'+ THRESHOLD_PERCENTAGE + " --" + ANIMAL + ".png" ), dpi = 300)
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



################# MAIN ################################

if __name__ == "__main__":
    print('Hello! You are running the Linear Regression Model for CMR!')


    if(sys.argv[1] == "mouse"):
        ANIMAL = "mouse"
        CHROMOSOMES = 19
        GC_CONTENT_PATH = './data/mouse/mouse_gc_content.csv'
        SINGLE_CHR_PRED_PATH = './data/mouse/PredictionResultsOnChr1/'
        CHART_OUTPUT_PATH = './output/mouse/linreg/'
        
        ALL_CHR_PREDICTIONS_PATH = './data/human/all_predictions/'
        SCATTERPLOTS_SAVE_PATH = './output/mouse/linreg/scatterplots/'
        BOXPLOTS_SAVE_PATH = './output/boxplots/mouse/'
        GCPLOT_SAVE_PATH = './output/gcerrorplots/mouse/'
        TRAININGLOSS_PATH = './data/mouse/trainingloss/00TrainLoss_sigmoid.txt'
        LOSS_ANALYSIS_PATH = './output/trainingloss_analysis/mouse/'
        DATA_OUTPUT_PATH = './output/mouse/linreg/mouse_linReg_Model_data.csv'
    
    elif(sys.argv[1] == "human"):
        ANIMAL = "human"
        CHROMOSOMES = 22
        GC_CONTENT_PATH = './data/human/human_gc_content.csv'
        SINGLE_CHR_PRED_PATH = './data/human/PredictionResultsOnChr1/00TestPredictions_sigmoid.txt'
        CHART_OUTPUT_PATH = './output/human/linreg/'
        
        ALL_CHR_PREDICTIONS_PATH = './data/human/all_predictions/'
        SCATTERPLOTS_SAVE_PATH = './data//scatterplots/human/'
        BOXPLOTS_SAVE_PATH = './output/boxplots/human/'
        GCPLOT_SAVE_PATH = './output/gcerrorplots/human/'
        TRAININGLOSS_PATH = './data/human/trainingloss/00TrainLoss_sigmoid.txt'
        LOSS_ANALYSIS_PATH = './output/trainingloss_analysis/human/'
        DATA_OUTPUT_PATH = './output/human/linreg/human_linReg_Model_data.csv'
    
    else:
        raise Exception("Sorry, please choose 'human' or 'mouse'")


    #GLOBAL:
    HIGH_GC = 0.45
    MID_GC = 0.40

    dataset = []
    zero_norms = []
    chromLengths = []
    binwiseData = []
    binwiseLabels = []
    mse_values = []

    #create a pandas dataframe so we can store our generated results:
    df_total = pd.DataFrame(columns=['Chromosome','Predictions','Targets','GC'])
    
    #get GC_content data
    gc_df3 = pd.read_csv(GC_CONTENT_PATH, sep = ",", header=None, comment= '#')
    gc_df3.columns = ['chromosome', 'start', 'end', 'A Portion', 'B Portion', 'GC Content', 'AT Content']

    for CHROM in range(1,CHROMOSOMES+1):
        #TEST DATA:
        df_temp = pd.DataFrame()
        test_wnull_df = gc_df3[gc_df3['chromosome'] == ('chr'+ str(CHROM))].copy()
        test_df = test_wnull_df[test_wnull_df['A Portion'].notnull()].copy()
        test_df = test_df.sample(frac=1).reset_index(drop=True) #shuffle test data
        x_test = test_df[['GC Content']].to_numpy()
        y_test = test_df[['A Portion']].to_numpy()

        #TRAIN DATA:
        df_index = gc_df3[gc_df3['chromosome'] == ('chr'+ str(CHROM))].index #gets all the indices of the test chromosome
        gc_df2 = gc_df3.drop(df_index)  #remove the test chromosome from the main df as we are using it for testing
        gc_df2 = gc_df2.sample(frac=1).reset_index(drop=True) #can shuffle the data if needed
        gc_df = gc_df2[gc_df2['A Portion'].notnull()].copy()
        x_train = gc_df[['GC Content']].to_numpy()
        y_train = gc_df[['A Portion']].to_numpy()        

        #GET THE HIGH MID LOW GC-CONTENT (y_test will have the same index as x_test later on, we rely on that)
        gc_content_list = []
        for bin in x_test:
            if bin >= HIGH_GC:
                gc_content_list.append('high')
            elif bin >=MID_GC:
                gc_content_list.append('mid')
            else:
                gc_content_list.append('low')


        #TRAIN THE MODEL:
        LR = LinearRegression()
        #print('...fitting')
        LR.fit(x_train, y_train)

        #y = mx + b
        b_val = LR.intercept_[0]
        m_val = LR.coef_[0][0]


        #PREDICT FOR THE TEST DATA:
        y_test_prediction = LR.predict(x_test)
        #CreatePredVsTargetChart(CHROM, y_test,y_test_prediction,gc_content_list)
        #CreateRegressionPerformanceChart(CHROM,x_test,y_test,y_test_prediction, 'GC Content (Test Data)', 'A-Compartment Preportion (Test Data)', 'Testing')


        #make the arrays 1D to pass to pearson
        squeezed_y = np.squeeze(y_test)
        squeezed_y_preds = np.squeeze(y_test_prediction)

        #get some stats about the predictions:
        rVal_spear, pVal_spear = getSpearman(squeezed_y,squeezed_y_preds)
        rVal_pearson, pVal_pearson = getPearson(squeezed_y,squeezed_y_preds)
        MSE_value = metrics.mean_squared_error(squeezed_y,squeezed_y_preds)
        mse_values.append(MSE_value)

        print('Chromosome: ', CHROM)
        print("The Spearman rho and pvalue: ",rVal_spear, "(",pVal_spear,")")
        print("   The Pearson r and pvalue: ",rVal_pearson, "(",pVal_pearson,")")
        print("                 The MSE is: ",MSE_value)
        print('*****************************************************************')

        #SETUP DATAFRAME:
        longListofCurrentChrom = []
        for x in range (len(squeezed_y)):
            longListofCurrentChrom.append('chr' + str(CHROM))

        df_temp['Chromosome'] = longListofCurrentChrom
        df_temp['Predictions'] = squeezed_y_preds.tolist()
        df_temp['Targets'] = squeezed_y.tolist()
        df_temp['GC'] = gc_content_list

        df_total = pd.concat([df_total,df_temp])
        
        
        #CONFUSION MATRIX:
        '''
        test_predictions = []
        for prediction in y_test_prediction:
            if prediction >= THRESHOLD:
                test_predictions.append(1)
            else:
                test_predictions.append(0)

        observed_values = []
        for label in y_test:
            if label >= THRESHOLD:
                observed_values.append(1)
            else:
                observed_values.append(0)

        ConfusionMatrix(observed_values,test_predictions)
        '''   
    
    print('The average MSE across all chromosomes is: ',mean(mse_values))
    print('The median MSE across all chromosomes is: ',np.percentile(mse_values,50))
    df_total.to_csv(DATA_OUTPUT_PATH, sep=',', encoding='utf-8')
