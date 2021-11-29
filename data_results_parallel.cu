#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <algorithm>
#include "cuda.h"
#include <limits.h>
#include "book.h"

//#include "gnuplot-iostream.h"
using namespace std;


const int NUM_PAIRS = 6;
const int DATA_LEN = 4;
const int NUM_VALUES = 6;

__global__ void calculate_correlations(float data[], float correlations[]) {
    int xIndex = 0;
    int yIndex = 0;
    int increment = DATA_LEN -1;
    int currIndex = threadIdx.x;

    //calculate the two indices of the data we are comparing using current thread
    while(currIndex >= (DATA_LEN -1)){
	xIndex++;
        increment--;
	if(increment < 1){
	     printf("Error calculating current Indexes to calculate correlations\n");
	     return;
	}
        currIndex -= increment;
    }
    xIndex *= NUM_VALUES;
    yIndex = (1 + currIndex) * NUM_VALUES;
    printf("hello from thread %d. I have xIndex %d and yIndex %d. Current increment is %d. Curr index %d\n", threadIdx.x, xIndex, yIndex, increment, currIndex);

    if(xIndex < 0 || xIndex >= DATA_LEN * NUM_VALUES || yIndex < 0 || yIndex >= DATA_LEN * NUM_VALUES || xIndex == yIndex){
	printf("Invalid indices calculated during correlation calculation function\n");
	return;
    }

    __syncthreads();
    if(threadIdx.x < NUM_PAIRS){
	    // Calculate mean of each dataset
	    float meanx = 0;
	    float meany = 0;
	    for (int i = 0; i < NUM_VALUES; i++) {
		meanx = meanx + data[xIndex + i];
		meany = meany + data[yIndex + i];
	    }
	    meanx = meanx / NUM_VALUES;
	    meany = meany / NUM_VALUES;

	    // Calculate deviation scores and product of deviation scores
	    float ssx = 0;
	    float ssy = 0;
	    float xy = 0;
	    for (int i = 0; i < NUM_VALUES; i++) {
		ssx = ssx + pow(data[xIndex + i] - meanx, 2);
		ssy = ssy + pow(data[yIndex + i] - meany, 2);
		xy = xy + (data[xIndex + i] - meanx) * (data[yIndex + i] - meany);
	    }

	    // Calculate correlation
	    correlations[threadIdx.x] = xy / sqrt(ssx * ssy);
	    __syncthreads();
    }
    else printf("Invalid thread number\n");
}


__global__ void display_correlations(float correlations[], char attributes[], int att_lengths[]){
	__syncthreads();
	//float correlation = correlations[threadIdx.x];
	int attributeAIndex = 0;
	int attributeBIndex = 0;
	int xIndex = 0;
	int yIndex = 0;
	int increment = DATA_LEN -1;
	int currIndex = threadIdx.x;
	
	while(currIndex >= DATA_LEN -1){
	    increment--;
	    xIndex++;
	    if(increment < 1){
	         printf("Error calculating current Indexes to display correlations\n");
	         return;
	    }
        currIndex -= increment;
        }
        xIndex += 1;
        yIndex = (2 + currIndex);
	for(int i =0; i <xIndex; i++) attributeAIndex += att_lengths[i];
	for(int i =0; i <yIndex; i++) attributeBIndex += att_lengths[i];

	if (abs(correlations[threadIdx.x]) > 1) {
		printf("Invalid correlation value. Exiting\n");
		return;
	}
	//__syncthreads();
	if(abs(correlations[threadIdx.x]) > 0.7){
		if(correlations[threadIdx.x] > 0) printf("Quantifiers %d and %d have a strong positive correlation of %f\n", xIndex, yIndex, correlations[threadIdx.x]);
		else printf("Quantifiers %d and %d have a strong negative correlation of %f\n", xIndex, yIndex, correlations[threadIdx.x]);
	}
	else if(abs(correlations[threadIdx.x]) > 0.5){
		if(correlations[threadIdx.x] > 0) printf("Quantifiers %d and %d have a moderate positive correlation of %f\n", xIndex, yIndex, correlations[threadIdx.x]);
		else printf("Quantifiers %d and %d have a moderate negative correlation of %f\n", xIndex, yIndex, correlations[threadIdx.x]);
	}
	else if(abs(correlations[threadIdx.x]) > 0.3){
		if(correlations[threadIdx.x] > 0) printf("Quantifiers %d and %d have a weak positive correlation of %f\n", xIndex, yIndex, correlations[threadIdx.x]);
		else printf("Quantifiers %d and %d have a weak negative correlation of %f\n", xIndex, yIndex, correlations[threadIdx.x]);
	}
	else printf("Quantifiers %d and %d have little-to-no correlation of %f\n", xIndex, yIndex, correlations[threadIdx.x]);
	return;
}


__global__ void calculate_linear_regressions(float correlations[], float data[])
{
    float sumx = 0;
    float sumy = 0;
    float sumxy = 0;
    float sumxSquared = 0;
    float sumySquared = 0;
    int xIndex = 0;
    int yIndex = 0;
    int increment = DATA_LEN -1;
    int currIndex = threadIdx.x;

    //calculate the two indices of the data we are comparing using current thread
    while(currIndex >= (DATA_LEN -1)){
	xIndex++;
        increment--;
	if(increment < 1){
	     printf("Error calculating current Indexes to calculate linear regressions\n");
	     return;
	}
        currIndex -= increment;
    }
    xIndex *= NUM_VALUES;
    yIndex = (1 + currIndex) * NUM_VALUES;
    
    if(abs(correlations[threadIdx.x]) < 0.3){
	printf("Minimal correlation between quanitifiers %d and %d. Skipping Calculating Regression.\n", xIndex / NUM_VALUES + 1, yIndex / NUM_VALUES + 1);
	return;
    }

    for (int i = 0; i < NUM_VALUES; i++) {
        sumx = sumx + data[xIndex + i];
        sumy = sumy + data[yIndex + i];
        sumxy = sumxy + data[xIndex + i] * data[yIndex + i];
        sumxSquared = sumxSquared + pow(data[xIndex + i], 2);
        sumySquared = sumySquared + pow(data[yIndex + i], 2);
    }

    float a = ((NUM_VALUES * sumxy) - (sumx * sumy)) / ((NUM_VALUES * sumxSquared) - pow(sumx, 2));
    float b = ((sumy * sumxSquared) - (sumx * sumxy)) / ((NUM_VALUES * sumxSquared) - pow(sumx, 2));
    printf("The calculated linear regression for quantifiers %d and %d is %fx + %f\n", xIndex / NUM_VALUES + 1, yIndex / NUM_VALUES + 1, a, b);
}


 
//flow of current tasks

// 1. Read Data (using mock data currently)
// 2. Calculate Correlations between each set of data
// 3. Categorize strength of correlations
// 4. Calculate Linear Regressions for non-minimal correlations
// 5. Plot data and regression for non-minimal correlations
int main()
{
    // test gnuplot on random data
    //demoGnuPlot();
    std::cout << "Beginning Display of Analyzed Data Results\n\n";

    // Mock attribute names
    string att_a = "Vaccinations / Million";
    string att_b = "Percent of Country Generally Open"; // Expect Postive Correlation with A
    string att_c = "Deaths / Million"; // Expect Negative Correlation with A
    string att_d = "Commute Time"; // No correlation with A

    // Definition of mock input arrays
    float input_data[DATA_LEN][6] = {{1, 2, 3 , 4, 5, 6}, { 5, 1, 2, 4, 8, 11}, { 20, 15, 11, 5, 7, 0 }, { 22, 21, 22, 21, 22, 21 }};
    string in_attributes[] = { att_a, att_b, att_c, att_d };

    float *correlations = new float[NUM_PAIRS];
    for(int i =0; i < NUM_PAIRS; i++) correlations[i] = 0;

    //convert 2d data to 1d array
    float data[DATA_LEN * NUM_VALUES];
    for(int i = 0; i < DATA_LEN; i++){
	for(int j = 0; j < NUM_VALUES; j++){
	    data[NUM_VALUES * i + j] = input_data[i][j];
	}
    }

    int NUM_CHARS = 0;
    int att_lengths[DATA_LEN];

    //calculate total length of attribute strings
    for(int i =0; i < DATA_LEN; i++){
	int j;
	for (j = 0; in_attributes[i][j] != '\0'; j++){
	NUM_CHARS += j;
 	att_lengths[i] = j;
	}
    }
    
    char attributes[NUM_CHARS];
    int temp =0;
    for(int i =0; i < DATA_LEN; i++){
	for(int j =0; j <= att_lengths[i]; j++){
	     attributes[temp] = in_attributes[i][j];
	     temp++;
	}
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Allocate device memory
    float *dev_correlations;
    float *dev_data;
    char *dev_attributes;
    int *dev_att_lengths;

    //Calculation of Correlations

    HANDLE_ERROR( cudaMalloc( (void**)&dev_data, NUM_VALUES * DATA_LEN * sizeof(float) ) );
    HANDLE_ERROR( cudaMemcpy( dev_data, data, NUM_VALUES * DATA_LEN * sizeof(float), cudaMemcpyHostToDevice) );  

    HANDLE_ERROR( cudaMalloc( (void**)&dev_correlations, NUM_PAIRS * sizeof(float) ) );
    HANDLE_ERROR( cudaMemcpy( dev_correlations, correlations, NUM_PAIRS * sizeof(float), cudaMemcpyHostToDevice) );  

    cout << "Calculating Correlations: \n\n";
    calculate_correlations<<<1, NUM_PAIRS>>>(dev_data, dev_correlations);
    cudaDeviceSynchronize();

    HANDLE_ERROR( cudaMemcpy(correlations, dev_correlations, NUM_PAIRS * sizeof(float), cudaMemcpyDeviceToHost));
    cout << "\n\nDisplaying Calculated Correlations: \n\n";

    for(int i =0; i < NUM_PAIRS; i++){
	cout << correlations[i] << " ";
    }
    cout << "\n\n";

    // Output Correlation Strengths

    HANDLE_ERROR( cudaMalloc( (void**)&dev_attributes, NUM_CHARS * sizeof(char) ) );
    HANDLE_ERROR( cudaMemcpy( dev_attributes, attributes, NUM_CHARS * sizeof(char), cudaMemcpyHostToDevice) );  
 
    HANDLE_ERROR( cudaMalloc( (void**)&dev_att_lengths, DATA_LEN * sizeof(int) ) );
    HANDLE_ERROR( cudaMemcpy( dev_att_lengths, att_lengths, DATA_LEN * sizeof(int), cudaMemcpyHostToDevice) ); 

    display_correlations<<<1, NUM_PAIRS>>>(dev_correlations, dev_attributes, dev_att_lengths);
    cudaDeviceSynchronize();

    // Calculate Linear Regressions
    cout << "\nCalulating Linear Regressions\n\n";
    calculate_linear_regressions<<<1, NUM_PAIRS>>>(dev_correlations, dev_data);
    cudaDeviceSynchronize();
    

    // get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    printf( "\n\nTime to calculate results:  %3.1f ms\n", elapsedTime );


    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );

    cudaFree ( dev_correlations);
    cudaFree (dev_data);
    cudaFree (dev_attributes);
    cudaFree(dev_att_lengths);

    cout << "\n\nSuccessfully Finished Execution :)";
    return 0;
}
