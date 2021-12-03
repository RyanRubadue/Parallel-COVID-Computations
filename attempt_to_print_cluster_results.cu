#include <iostream>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <bits/stdc++.h>
#include "cuda.h"
#include "../common/book.h"

using namespace std;

/*
	Column #: Data
	4: total_cases
	5: new_cases
	7: total_deaths
	8: new_deaths
	10: total_cases_per_million
	11: new_cases_per_million
	13: total_deaths_per_million
	14: new_deaths_per_million
	17: icu_patients
	18: icu_patients_per_million
	19: hosp_patients
	20: hosp_patients_per_million
	25: new_tests
	26: total_tests
	31: positive_rate
	32: tests_per_case
	34: total_vaccinations
	35: people_vaccinated
	36: people_fully_vaccinated
	37: total_boosters
	38: new_vaccinations
	48: median_age
*/

const string FILE_NAME = "Data/owid-covid-data-2021-11-03.csv"; // File location
const int COLUMN_NUMBER_1 = 4; // Column #
const int COLUMN_NUMBER_2 = 7; // Column #
const int COLUMN_NUMBER_3 = 31; // Column #
const int COLUMN_NUMBER_4 = 35; // Column #
const int NUM_RECORDS = 121955; // Number of Records to read
const int NUM_CLUSTERS = 5;
const int NUM_PAIRS = 6; //Number of possible Column pairings
const int NUM_COLUMNS = 4; //Number of columns we are using

void build_input(float input_1[], float input_2[], float input_3[], float input_4[], int locations[]){
	ifstream file;
	int lines = -1, location = 0, column_count = 0;
	string line, current_location, value;
	file.open(FILE_NAME);
	while (getline(file, line)) 
	{
		lines++;
		if (lines == 0 || lines-1 >= NUM_RECORDS){
			continue;
		}
		column_count = 0;
		stringstream s (line);
		while (getline(s, value, ','))
		{
			if (column_count == 2 && current_location != value){
				location++;
				current_location = value;
				locations[lines - 1] = location;
				cout << "Location #: " << location << "\t| Location: " << current_location << endl;
			}
			else if (column_count == 2 && current_location == value)
			{
				locations[lines - 1] = location;
			}
			else if (column_count == COLUMN_NUMBER_1) 
			{
				input_1[lines - 1] = value.empty() ? 0 : stof(value);
			}
			else if (column_count == COLUMN_NUMBER_2) 
			{
				input_2[lines - 1] = value.empty() ? 0 : stof(value);
			}
			else if (column_count == COLUMN_NUMBER_3) 
			{
				input_3[lines - 1] = value.empty() ? 0 : stof(value);
			}
			else if (column_count == COLUMN_NUMBER_4) 
			{
				input_4[lines - 1] = value.empty() ? 0 : stof(value);
			}
			column_count++;
		}
	}
}

void find_range(float x[], float centroids[], int* max_loc){
	
	//srand(time(0));
	for(int i =0; i < NUM_CLUSTERS; i++){
		int j = rand() % *max_loc; 
		centroids[i] = x[j];
	}
}
__global__ void calculate_centers(float data[], int clusters[], float centers[], int* max_loc){
	__shared__ int counts[NUM_CLUSTERS];
	__shared__ unsigned int temp_counts[NUM_CLUSTERS];
	
	__shared__ float temp_centers[NUM_CLUSTERS];
	
	if(threadIdx.x < NUM_CLUSTERS){
		temp_counts[threadIdx.x] = 0;
		counts[threadIdx.x] = 0;
		temp_centers[threadIdx.x] = 0.0;
		centers[threadIdx.x] = 0.0;
	}
 
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	
	__syncthreads();
	
	while(i < *max_loc){
		atomicAdd(&temp_centers[clusters[i]], data[i]);
		atomicAdd(&temp_counts[clusters[i]], 1);
		i += offset;
	}

	__syncthreads();
	i = threadIdx.x;
	
	if (i < NUM_CLUSTERS){
		atomicAdd(&centers[threadIdx.x], temp_centers[threadIdx.x]);
		atomicAdd(&counts[threadIdx.x], temp_counts[threadIdx.x]);
	}
	
	i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (i < NUM_CLUSTERS){
		if(counts[i] != 0){
			centers[i] = centers[i] / counts[i];
		}
	}
}

__global__ void compare(float data[], float centers[], int clusters[], bool* change_cluster, int* max_loc) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;	
	
	float min_diff = abs(data[i]-centers[clusters[i]]);

	while (i < *max_loc ){ 
		for(int j = 0; j < NUM_CLUSTERS; j++){
			float diff = abs(data[i] - centers[j]);
			if (diff < min_diff){
				min_diff = diff;
				
				if( (clusters[i] != j) && (*change_cluster == false) ){
					*change_cluster = true;
				}
				
				clusters[i] = j;
			}
			
			if (abs(min_diff - 0) < 0.0001){
				break;
			}
		}
		i += offset;
	}
}

__global__ void display_data_averages(float data_avg[], int locations[], float data[], int country_count[], float country_avg[], int* max_loc){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	while ( i < NUM_RECORDS){
		atomicAdd(&data_avg[locations[i]], data[i]);
		atomicAdd(&country_count[locations[i]], 1);	
		i = i + offset;
	}
	__syncthreads();

	i = threadIdx.x + blockIdx.x * blockDim.x;
	while (i < *max_loc){
	    country_avg[i] = data_avg[i] / country_count[i];
	    i = i + offset;
	}
}

__global__ void calculate_correlations(float result_data[], float correlations[]) {
	int xIndex = 0;
	int yIndex = 0;
	int increment = NUM_COLUMNS -1;
	int currIndex = threadIdx.x;

	//calculate the two indices of the data we are comparing using current thread
	while(currIndex >= (NUM_COLUMNS -1)){
	xIndex++;
		increment--;
	if(increment < 1){
		 printf("Error calculating current Indexes to calculate correlations\n");
		 return;
	}
		currIndex -= increment;
	}
	xIndex *= NUM_RECORDS;
	yIndex = (1 + currIndex) * NUM_RECORDS;
	//printf("hello from thread %d. I have xIndex %d and yIndex %d. Current increment is %d. Curr index %d\n", threadIdx.x, xIndex, yIndex, increment, currIndex);

	if(xIndex < 0 || xIndex >= NUM_COLUMNS * NUM_RECORDS || yIndex < 0 || yIndex >= NUM_COLUMNS * NUM_RECORDS || xIndex == yIndex){
	printf("Invalid indices calculated during correlation calculation function\n");
	return;
	}

	__syncthreads();
	if(threadIdx.x < NUM_PAIRS){
		// Calculate mean of each dataset
		float meanx = 0;
		float meany = 0;
		for (int i = 0; i < NUM_RECORDS; i++) {
		meanx = meanx + 0.0001 * result_data[xIndex + i];
		meany = meany + 0.0001 * result_data[yIndex + i];
		}
		meanx = meanx / (NUM_RECORDS * 0.0001);
		meany = meany / (NUM_RECORDS * 0.0001);
		// Calculate deviation scores and product of deviation scores
		float ssx = 0;
		float ssy = 0;
		float xy = 0;
		for (int i = 0; i < NUM_RECORDS; i++) {
		ssx = ssx + 0.0001 * pow(result_data[xIndex + i] - meanx, 2);
		ssy = ssy + 0.0001 * pow(result_data[yIndex + i] - meany, 2);
		xy = xy + 0.0001 * (result_data[xIndex + i] - meanx) * (result_data[yIndex + i] - meany);
		}

		// Calculate correlation
		correlations[threadIdx.x] = (xy / sqrt(ssx * ssy));
		__syncthreads();
	}
	else printf("Invalid thread number\n");
}


__global__ void display_correlations(float correlations[]){
	__syncthreads();
	//float correlation = correlations[threadIdx.x];
	int xIndex = 0;
	int yIndex = 0;
	int increment = NUM_COLUMNS -1;
	int currIndex = threadIdx.x;
	
	while(currIndex >= NUM_COLUMNS -1){
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

	if (abs(correlations[threadIdx.x]) > 1) {
		printf("Invalid correlation value. Exiting\n");
		return;
	}
	//__syncthreads();
	if(abs(correlations[threadIdx.x]) > 0.7){
		if(correlations[threadIdx.x] > 0) printf("Columns %d and %d have a strong positive correlation of %f\n", xIndex, yIndex, correlations[threadIdx.x]);
		else printf("Columns %d and %d have a strong negative correlation of %f\n", xIndex, yIndex, correlations[threadIdx.x]);
	}
	else if(abs(correlations[threadIdx.x]) > 0.5){
		if(correlations[threadIdx.x] > 0) printf("Columns %d and %d have a moderate positive correlation of %f\n", xIndex, yIndex, correlations[threadIdx.x]);
		else printf("Columns %d and %d have a moderate negative correlation of %f\n", xIndex, yIndex, correlations[threadIdx.x]);
	}
	else if(abs(correlations[threadIdx.x]) > 0.3){
		if(correlations[threadIdx.x] > 0) printf("Columns %d and %d have a weak positive correlation of %f\n", xIndex, yIndex, correlations[threadIdx.x]);
		else printf("Columns %d and %d have a weak negative correlation of %f\n", xIndex, yIndex, correlations[threadIdx.x]);
	}
	else printf("Columns %d and %d have little-to-no correlation of %f\n", xIndex, yIndex, correlations[threadIdx.x]);
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
	int increment = NUM_COLUMNS -1;
	int currIndex = threadIdx.x;

	//calculate the two indices of the data we are comparing using current thread
	while(currIndex >= (NUM_COLUMNS -1)){
	xIndex++;
		increment--;
	if(increment < 1){
		 printf("Error calculating current Indexes to calculate linear regressions\n");
		 return;
	}
		currIndex -= increment;
	}
	xIndex *= NUM_RECORDS;
	yIndex = (1 + currIndex) * NUM_RECORDS;
	
	if(abs(correlations[threadIdx.x]) < 0.3){
	printf("Minimal correlation between quanitifiers %d and %d. Skipping Calculating Regression.\n", xIndex / NUM_RECORDS + 1, yIndex / NUM_RECORDS + 1);
	return;
	}

	for (int i = 0; i < NUM_RECORDS; i++) {
		sumx = sumx + 0.0001 * data[xIndex + i];
		sumy = sumy + 0.0001 * data[yIndex + i];
		sumxy = sumxy + 0.0001 * (data[xIndex + i] * data[yIndex + i]);
		sumxSquared = sumxSquared + 0.0001 * pow(data[xIndex + i], 2);
		sumySquared = sumySquared + 0.0001 * pow(data[yIndex + i], 2);
	}
	float a = 10000 * (((NUM_RECORDS * sumxy) - (sumx * sumy)) / ((NUM_RECORDS * sumxSquared) - pow(sumx, 2)));
	float b = 10000 * (((sumy * sumxSquared) - (sumx * sumxy)) / ((NUM_RECORDS * sumxSquared) - pow(sumx, 2)));
	printf("The calculated linear regression for columns %d and %d is %fx + %f\n", xIndex / NUM_RECORDS + 1, yIndex / NUM_RECORDS + 1, a, b);
}

int main() {
	cout << "Starting..." << endl;
	
	 
	float* centers = new float[NUM_CLUSTERS];
	float* input_1 = new float[NUM_RECORDS];
	float* input_2 = new float[NUM_RECORDS];
	float* input_3 = new float[NUM_RECORDS];
	float* input_4 = new float[NUM_RECORDS];
	int* locations = new int[NUM_RECORDS];
	
	
	bool* change_clusters = new bool(true);
	int counter = 20;
	float elapsedTime;

	float *dev_data;
	float *dev_centers;
	int *dev_clusters;
	int* dev_max_loc;
	bool *dev_change_clusters;
	int* dev_locations;
	int* dev_country_count;
	float* dev_data_avg;
	float* dev_country_avg;

	int* index = new int;
	int* mapping = new int[5];

	cudaEvent_t	 start, stop;
	HANDLE_ERROR( cudaEventCreate( &start ) );
	HANDLE_ERROR( cudaEventCreate( &stop ) );
	HANDLE_ERROR( cudaEventRecord( start, 0 ) );

	build_input(input_1, input_2, input_3, input_4, locations);
	
	int* max_loc = max_element(locations,locations+NUM_RECORDS);
	int* clusters = new int[*max_loc];
	float* country_avg = new float[*max_loc];
	int* country_count = new int[*max_loc];
	
	float* data_avg = new float[*max_loc];
	const int SIZE_F = *max_loc * sizeof(float);
	const int SIZE_R = NUM_RECORDS * sizeof(float);

	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
	printf( "Time to Read:  %3.1f ms\n", elapsedTime );
	HANDLE_ERROR( cudaEventRecord( start, 0 ) );

	HANDLE_ERROR( cudaMalloc( (void**)&dev_data, NUM_RECORDS*sizeof(float) ) );
	HANDLE_ERROR( cudaMemcpy( dev_data, input_1, NUM_RECORDS*sizeof(float), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_data_avg, SIZE_F) );
	HANDLE_ERROR( cudaMemset( dev_data_avg, 0, SIZE_F) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_country_count, *max_loc*sizeof(int) ) );
	HANDLE_ERROR( cudaMemset( dev_country_count, 0, *max_loc*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc((void**)&dev_locations, NUM_RECORDS*sizeof(int)) );
	HANDLE_ERROR( cudaMemcpy(dev_locations, locations, NUM_RECORDS*sizeof(int), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMalloc((void**)&dev_country_avg, SIZE_F) );
	HANDLE_ERROR( cudaMemset(dev_country_avg, 0, SIZE_F) );
	HANDLE_ERROR( cudaMalloc((void**)&dev_max_loc, sizeof(int)) );
	HANDLE_ERROR( cudaMemcpy(dev_max_loc, max_loc, sizeof(int), cudaMemcpyHostToDevice));
	
	display_data_averages<<<1, 1024>>>(dev_data_avg, dev_locations, dev_data, dev_country_count, dev_country_avg, dev_max_loc);
	
	HANDLE_ERROR( cudaMemcpy(country_avg, dev_country_avg, *max_loc * sizeof(float), cudaMemcpyDeviceToHost) );
	

	find_range(country_avg, centers, max_loc);
	cout << "Original centers:";
	for(int i = 0; i < NUM_CLUSTERS; i++){
		cout << centers[i] << " ";
	}
	cout << endl;
	
	HANDLE_ERROR( cudaMalloc( (void**)&dev_centers, NUM_CLUSTERS * sizeof( float ) ) );
	HANDLE_ERROR( cudaMemcpy( dev_centers, centers, NUM_CLUSTERS*sizeof(float), cudaMemcpyHostToDevice ) ); 
	HANDLE_ERROR( cudaMalloc( (void**)&dev_clusters, *max_loc * sizeof( int ) ) );
	HANDLE_ERROR( cudaMemset( dev_clusters, 0, *max_loc * sizeof( int )) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_change_clusters, sizeof(bool)));
	HANDLE_ERROR( cudaMemset( dev_change_clusters, false, sizeof(bool)) );
							  
	compare<<<1,256>>>(dev_country_avg,dev_centers,dev_clusters,dev_change_clusters,dev_max_loc);

	while( (*change_clusters == true) && (counter > 0) ){
		HANDLE_ERROR( cudaMemset( dev_change_clusters, false, sizeof(bool)) );
		
		calculate_centers<<<1,5>>>(dev_country_avg, dev_clusters, dev_centers, dev_max_loc);
		
		HANDLE_ERROR( cudaMemcpy(centers, dev_centers, NUM_CLUSTERS*sizeof(float), cudaMemcpyDeviceToHost) );
		cout << "Centers: ";
		for(int i = 0; i < NUM_CLUSTERS; i++){
			cout << centers[i] << " ";
		}
		cout << endl;
		
		compare<<<1,20>>>(dev_country_avg,dev_centers,dev_clusters,dev_change_clusters, dev_max_loc);
		
		HANDLE_ERROR( cudaMemcpy( change_clusters, dev_change_clusters, sizeof(bool), cudaMemcpyDeviceToHost) );
		//cout << "Change_clusters is: " << *change_clusters << endl;
		counter--;
	}
	

	HANDLE_ERROR( cudaMemcpy(clusters,dev_clusters, *max_loc*sizeof(int), cudaMemcpyDeviceToHost) );
	
	cout << "Final clusters: " << endl;
	for(int i = 0; i < *max_loc; i++){
		cout << clusters[i] << " ";
	}
	cout << endl;

	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
	printf( "Time to Analyze:  %3.1f ms\n", elapsedTime );

	int* cluster_count = new int[NUM_CLUSTERS];
	for(int i = 0; i < NUM_CLUSTERS; i++){
		cluster_count[i] = 0;
	}
	for(int i = 0; i < *max_loc; i++){
			cluster_count[clusters[i]]++;
		}

	for(int i = 0; i < NUM_CLUSTERS; i++){
		cout << "Cluster " << i << ": " << cluster_count[i] << endl;
	}
	
	//Results - Ryan

	cout << "Beginning Display of Results\n\n";

	// Allocate device memory
	float *dev_correlations;
	float *dev_result_data;

	float result_data[SIZE_R * NUM_COLUMNS];
	float *correlations = new float[NUM_PAIRS];

	for(int i =0; i < *max_loc; i++){
		if(country_count[i] < 1 || data_avg[i] != data_avg[i]) cout << "No entries seen for country " << i << "\n";
		else cout << "Cluster Average for Country " << i << ":   " << abs(data_avg[i] / country_count[i]) << "\n";
	}
	
	for(int i =0; i < NUM_PAIRS; i++) correlations[i] = 0;
	for(int i = 0; i < NUM_RECORDS; i++){
	    result_data[i] = input_1[i];
    	    result_data[i + NUM_RECORDS] = input_2[i];
	    result_data[i + 2*NUM_RECORDS] = input_3[i];
	    result_data[i+3*NUM_RECORDS] = input_4[i];
	}

	HANDLE_ERROR( cudaMalloc( (void**)&dev_result_data, SIZE_R * NUM_COLUMNS ) );
	HANDLE_ERROR( cudaMemcpy( dev_result_data, result_data, SIZE_R * NUM_COLUMNS, cudaMemcpyHostToDevice) );  

	HANDLE_ERROR( cudaMalloc( (void**)&dev_correlations, NUM_PAIRS * sizeof(float) ) );
	HANDLE_ERROR( cudaMemcpy( dev_correlations, correlations, NUM_PAIRS * sizeof(float), cudaMemcpyHostToDevice) );  

	cout << "\n\nCalculating Correlations... ";
	calculate_correlations<<<1, NUM_PAIRS>>>(dev_result_data, dev_correlations);
	cudaDeviceSynchronize();

	HANDLE_ERROR( cudaMemcpy(correlations, dev_correlations, NUM_PAIRS * sizeof(float), cudaMemcpyDeviceToHost));
	cout << "\n\nDisplaying Calculated Correlations: \n\n";

	// Output Correlation Strengths

	display_correlations<<<1, NUM_PAIRS>>>(dev_correlations);
	cudaDeviceSynchronize();

	//Calculate Linear Regressions
	cout << "\nCalulating Linear Regressions\n\n";
	calculate_linear_regressions<<<1, NUM_PAIRS>>>(dev_correlations, dev_result_data);
	cudaDeviceSynchronize();


	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
	printf( "\nTime to Output:  %3.1f ms\n", elapsedTime );

	HANDLE_ERROR( cudaEventDestroy( start ) );
	HANDLE_ERROR( cudaEventDestroy( stop ) );
	cudaFree( dev_data );
	cudaFree( dev_centers );
	cudaFree( dev_clusters );
	cudaFree( dev_change_clusters );
	cudaFree( dev_max_loc );
	cudaFree ( dev_correlations);
	cudaFree (dev_data);
	cudaFree(dev_country_count);
	
	return 0;
}
