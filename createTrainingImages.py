# This is a sample Python script.
import numpy as np
import cv2
import pywt
import pywt.data
import matplotlib.pyplot as plt
import os 
import time


samplingFreq = 100
dataPeriod = 6 # sec
numSamples = samplingFreq*dataPeriod*3
numSamplesCh = samplingFreq*dataPeriod
stepSize = 129
maxSteps = 15
minSteps = 0
sampThresh = 100  # number of samples to be ignored at the start and end of every coef array
minThreshRpeakCh1 = 5
minThreshRpeakCh2 = minThreshRpeakCh1
minThreshRpeakCh3 = minThreshRpeakCh1 - 2


ipfolder = r'D:\TEJAS\Projects\ECG\Training Images\File'
opfolder = r'D:\TEJAS\Projects\ECG\Training Images\Output1'
#filename = 'Mr.SURENDRA SINGH 45Y M 104 15-08-2022 11-00 hrs.ecg'

for filename in os.listdir(ipfolder):
    # Check if the file is a valid ECG file
    if filename.endswith(".ecg"):
        # Create output folder based on the file name
        output_folder = os.path.join(opfolder, os.path.splitext(filename)[0])

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Create a subfolder inside the output folder for the current file
        result_subfolder = os.path.join(output_folder, "Results")
        if not os.path.exists(result_subfolder):
            os.makedirs(result_subfolder)
        readDataFromFile = open(os.path.join(ipfolder,filename),'rb')
        datatype = np.dtype('B')
        filedata = np.fromfile(readDataFromFile, datatype)
        #print(filedata)
        p = list(filedata)
        ecgarr = np.array(p)
        totSamples = len(ecgarr)
        numRows = int(totSamples/numSamples)
        remainderSamples = totSamples % numSamples
        curateData = [[0] * numSamples for i in range(numRows)]
        overlapDataMini = []
        overlapData = []
        totEntries = int(totSamples/3)
        totChEntries = int(numSamples/3)

        opfolder = result_subfolder

        fch1 = [0 for i in range(totEntries)]
        fch2 = [0 for i in range(totEntries)]
        fch3 = [0 for i in range(totEntries)]
        # Divide the total samples in sets of 'numSamples'
        xcntr = 0
        for x in range(0,(totSamples-remainderSamples),numSamples):
            ycntr = 0
            for y in range(x,(x+numSamples),1):
                #print(y)
                curateData[xcntr][ycntr] = ecgarr[y]
                ycntr = ycntr + 1

            xcntr = xcntr + 1

        # Separating 3 channels
        cntr = 0
        for y in range(1,totEntries,1):
            #for y in range(1,(numSamples),1):
                #print(y)
            fch1[cntr] = ecgarr[3 * (y - 1)]
            fch2[cntr] = ecgarr[3 * (y - 1) + 1]
            fch3[cntr] = ecgarr[3 * (y - 1) + 2]

            cntr = cntr + 1

        prDiff = 7
        rtDiff = 7
        pqrInterval = 20
        rstInterval = 20
        maxRPeaks = 9 # Calculate median every such R peaks

        # Creating overlap arrays as per the time epoch - keep this number odd
        x = 0
        cnt = 0
        itr = 0
        ch1Arr = [0 for i in range(totChEntries)]
        ch1ArrNoise = [0 for i in range(totChEntries)]
        ch2Arr = [0 for i in range(totChEntries)]
        ch3Arr = [0 for i in range(totChEntries)]


        ###########################################################################################################################

        ## Function to define channel analysis
        def channelAnalysisImg(fch, stepSize, numSamplesCh, sampThresh, maxSteps, minSteps, numSamples,
                            filename, numRow, chNum, startSamp, endSamp, itr, opfolder):
            coef, freqs = pywt.cwt(fch, np.arange(1, stepSize), 'morl')

            fig, axs = plt.subplots(2)
            fig.suptitle('Raw signal, CWT coef 0')  #### Add a dynamic name to save the plot
            axs[0].plot(fch)
            axs[1].plot(coef[0,:])

            nameWord = str(chNum) + '_'
            nameWord = nameWord + str(itr)

            pltword = '_plot.png'
            tempname = nameWord + pltword
            tempname = filename + tempname
            pltfilename = os.path.join(opfolder, tempname)
            plt.savefig(pltfilename, dpi=300, bbox_inches='tight')
            plt.close()


            cntr = 0
            coefArr = [0 for i in range(numSamplesCh)]
            RpeakArr = [0 for i in range(numSamplesCh)]
            max = 0
            for cntr in range(numSamplesCh):
                coefArr[cntr] = coef[0, cntr]

            # Create the image
            minImg = 0
            maxImg = 0
            row = 0
            col = sampThresh
            img = np.zeros([(maxSteps - minSteps), (numSamplesCh - (2 * sampThresh)), 3], dtype=np.uint8)
            ### Convert the coeffient matrix of CWT into an image
            for row in range(minSteps, maxSteps, 1):
                for col in range(sampThresh, (numSamplesCh - sampThresh), 1):
                    val = coef[row, col]
                    # img[row,col] = val
                    if val < minImg:
                        minImg = val
                    if val > maxImg:
                        maxImg = val

            row = 0
            col = sampThresh
            for row in range(maxSteps - minSteps):
                for col in range(numSamplesCh - (2 * sampThresh)):
                    oldVal = coef[row, col]
                    num = (oldVal - minImg) * 255
                    den = (maxImg - minImg)
                    newVal = num / den
                    if newVal > 255:
                        newVal = 255
                    elif newVal < 0:
                        newVal = 0
                    img[row, col] = newVal

            ipword = '_predict_ip.bmp'
            tempname = nameWord + ipword

            tempname = filename + tempname
            ipfilename = os.path.join(opfolder, tempname)
            cv2.imwrite(ipfilename, img)  #### Change to dynamic name

            return tempname

        ###########################################################################################################################

        while (x < (totEntries-numSamplesCh)):
            ## Create a median buffer array for every epoch of 6 sec
            medBuffArr = [[0] * (pqrInterval+rstInterval+1) for i in range(maxRPeaks)]
            medBufNew = [[0] * maxRPeaks for i in range(pqrInterval + rstInterval + 1)]
            median = [0 for i in range(pqrInterval + rstInterval + 1)]
            cnt = 0
            itr = itr + 1
            #print(itr)
            #if itr == 1491:
            #    print("Debug")
            startSamp = x
            endSamp = (x+numSamplesCh)
            for x in range(x,(x+numSamplesCh),1):
                ch1Arr[cnt] = fch1[x]
                ch2Arr[cnt] = fch2[x]
                ch3Arr[cnt] = fch3[x]
                cnt = cnt + 1

            start_time = time.time()

            ## Multiplying the channel data array with a noise constant
            #noise = constNoise*noise
            #ch1ArrNoise = ch1Arr + noise

            #fig, axs = plt.subplots(2)
            #fig.suptitle('Raw signal, Raw signal with noise')  #### Add a dynamic name to save the plot
            #axs[0].plot(ch1Arr)
            #axs[1].plot(ch1ArrNoise)

            ######################################################################################################
            ##Analyze the channel
            ## Channel no. 1
            meanArr = np.mean(ch1Arr)
            chNum = 1
            categoryID1 = 0 # reset
            sampCntArr1 = [0 for i in range(numSamplesCh)]
            heartRateArr1 = [0 for i in range(numSamplesCh)]
            tempname1 = channelAnalysisImg(ch1Arr, stepSize, numSamplesCh, sampThresh, maxSteps, minSteps,
                                                                                        numSamples, filename, numRows,
                                                                                         chNum, startSamp, endSamp,
                                                                                    itr, opfolder)

            tempname2 = channelAnalysisImg(ch2Arr, stepSize, numSamplesCh, sampThresh,
                                                                                                       maxSteps, minSteps,
                                                                                                       numSamples,
                                                                                                       filename, numRows,
                                                                                                       chNum, startSamp,
                                                                                                       endSamp, itr,
                                                                                                       opfolder)

            tempname3 = channelAnalysisImg(ch3Arr, stepSize,
                                                                                                       numSamplesCh,
                                                                                                       sampThresh,
                                                                                                       maxSteps, minSteps,
                                                                                                       numSamples,
                                                                                                       filename, numRows,
                                                                                                       chNum, startSamp,
                                                                                                       endSamp,
                                                                                                       itr,
                                                                                                       opfolder)

            end_time = time.time()
            calcTime = end_time - start_time
            print("CalcTime: %.2f" % calcTime)

            x = x - (sampThresh * 2)  ## Reset x value to compensate for the overlap

        complete_img = np.zeros((100, 300, 3), dtype=np.uint8)  # Example image dimensions
        cv2.putText(complete_img, "Complete File Run", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        complete_filename = os.path.join(output_folder, "complete_file_run.png")
        cv2.imwrite(complete_filename, complete_img)
        print(f"Completed processing file: {filename}")

