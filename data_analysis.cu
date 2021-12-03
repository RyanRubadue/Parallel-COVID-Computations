#include <iostream>
#include <cmath>
#include <algorithm>
#include "cuda.h"
#include "book.h"

using namespace std;

const int NUM_RECORDS = 20;
const int NUM_CLUSTERS = 5;

void find_range(float x[], float centroids[]){
    float* min = min_element(x,x+NUM_RECORDS);
    float* max = max_element(x,x+NUM_RECORDS);
    float range = *max - *min;
    float group_size = range / NUM_CLUSTERS;
    for(int i =0; i < NUM_CLUSTERS; i++){
        centroids[i] = *min + group_size * i;
    }
}

__global__ void calculate_centers(float data[], int clusters[], float centers[]){
    
    __shared__ int counts[NUM_CLUSTERS];
    __shared__ float x[NUM_RECORDS];
    __shared__ int shared_clusters[NUM_RECORDS];
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
    
    /*
    if(i < NUM_RECORDS){
        x[i] = data[i];
        shared_clusters[i] = clusters[i];
    }
    */
    
    /*
    if (i < NUM_CLUSTERS){
        centers[i] = 0.0;
        counts[i] = 0;
    }
    */
    
    __syncthreads();
    
    while(i < NUM_RECORDS){
        atomicAdd(&temp_centers[clusters[i]], data[i]);
        atomicAdd(&temp_counts[clusters[i]], 1);
        i += offset;
    }

    __syncthreads();
    i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < NUM_CLUSTERS){
        atomicAdd(&centers[threadIdx.x], temp_centers[threadIdx.x]);
        atomicAdd(&counts[threadIdx.x], temp_counts[threadIdx.x]);
    }
    if (i < NUM_CLUSTERS){
        if(counts[i] != 0){
            centers[i] = centers[i] / counts[i];
        }
    }
}

__global__ void compare(float data[], float centers[], int clusters[], bool* change_cluster) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    

    float min_diff = abs(data[i]-centers[clusters[i]]);

    if (i < NUM_RECORDS ){
        for(int j = 0; j < NUM_CLUSTERS; j++){
            float diff = abs(data[i] - centers[j]);
            if (diff < min_diff){
                min_diff = diff;
                
                /****NEED TO LOCK?****
                if( (clusters[i] != j) && (*change_cluster == false) ){
                    *change_cluster = true;
                }
                *********************/
                
                clusters[i] = j;
            }
            
            if (abs(min_diff - 0) < 0.0001){
                break;
            }
        }
    }
}

__global__ void classify(int clusters[], int locations[], int mapping[], int* max_loc, int* index){
    
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    *index = 0;
    if (i < NUM_RECORDS){
        for(int j = 0; j <= *max_loc; j++){
            if(i < NUM_CLUSTERS){
                mapping[i] = 0;
            }
            
            if(locations[i] == j){
                int x = clusters[i];
                atomicAdd(&mapping[x], 1);
            }
            
            __syncthreads();
            
            if (i == 1){
                int max = mapping[0];
                *index = 0;

                for(int p = 1; p < NUM_CLUSTERS; p++){
                    if (mapping[p] > max){
                        max = mapping[p];
                        *index = p;
                    }
                }
            }
            __syncthreads();
            if(locations[i] == j){
                clusters[i] = *index;
            }
        }
    }
}    
    


int main() {

    const int SIZE_F = NUM_RECORDS * sizeof(float); 

    // capture the start time
    // starting the timer here so that we include the cost of
    // all of the operations on the GPU.  if the data were
    // already on the GPU and we just timed the kernel
    // the timing would drop from 74 ms to 15 ms.  Very fast.
    float input[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};

    float* centers = new float[NUM_CLUSTERS];
    

    int counter = 20;
    
    int* locations = new int[NUM_RECORDS];
    cout << "Locations: ";
    locations[0] = 1;
    locations[1] = 5;
    locations[2] = 2;
    locations[3] = 4;
    locations[4] = 5;
    locations[5] = 1;
    locations[6] = 3;
    locations[7] = 2;
    locations[8] = 3;
    locations[9] = 4;
    locations[10] = 1;
    locations[11] = 1;
    locations[12] = 3;
    locations[13] = 5;
    locations[14] = 4;
    locations[15] = 1;
    locations[16] = 2;
    locations[17] = 2;
    locations[18] = 3;
    locations[19] = 5;
    
    for(int i = 0; i < NUM_RECORDS; i++){
        //locations[i] = rand() % 10;
        cout << locations[i] << " ";
    }
    cout << endl;
    int* max_loc = max_element(locations,locations+NUM_RECORDS);
    
    int* clusters = new int[NUM_RECORDS];
    
    find_range(input, centers);
    cout << "Original centers:";
    for(int i = 0; i < NUM_CLUSTERS; i++){
        cout << centers[i] << " ";
    }
    cout << endl;
    
    bool* change_clusters = new bool(true);
    
    
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    // allocate memory on the GPU for the file's data
    float *dev_data;
    float *dev_centers;
    int *dev_clusters;
    int* dev_max_loc;
    bool *dev_change_clusters;
    int* dev_mapping;
    int* dev_locations;
    int* dev_index;
    
    HANDLE_ERROR( cudaMalloc( (void**)&dev_data, SIZE_F ) );
    HANDLE_ERROR( cudaMemcpy( dev_data, input, SIZE_F,
                              cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMalloc( (void**)&dev_centers,
                              NUM_CLUSTERS * sizeof( float ) ) );
    HANDLE_ERROR( cudaMemcpy( dev_centers, centers, NUM_CLUSTERS*sizeof(float),
                              cudaMemcpyHostToDevice ) );
                              
    HANDLE_ERROR( cudaMalloc( (void**)&dev_clusters,
                              NUM_RECORDS * sizeof( int ) ) );
    HANDLE_ERROR( cudaMemset( dev_clusters, 0,
                              NUM_RECORDS * sizeof( int ) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_change_clusters, sizeof(bool)));
    
    HANDLE_ERROR( cudaMemset( dev_change_clusters, false, sizeof(bool)) );
                              
    compare<<<1,20>>>(dev_data,dev_centers,dev_clusters,dev_change_clusters);
    
    HANDLE_ERROR( cudaMemcpy(clusters, dev_clusters, NUM_RECORDS*sizeof(int), cudaMemcpyDeviceToHost));
    cout << "Clusters: ";
        for(int i = 0; i < NUM_RECORDS; i++){
            cout << clusters[i] << " ";
        }
        cout << endl;
    
    while( (*change_clusters == true) && (counter > 0) ){
        HANDLE_ERROR( cudaMemset( dev_change_clusters, false, sizeof(bool)) );
        calculate_centers<<<1,5>>>(dev_data, dev_clusters, dev_centers);
        HANDLE_ERROR( cudaMemcpy(centers, dev_centers, NUM_CLUSTERS*sizeof(float), cudaMemcpyDeviceToHost) );
        cout << "Centers: ";
        for(int i = 0; i < NUM_CLUSTERS; i++){
            cout << centers[i] << " ";
        }
        cout << endl;
        compare<<<1,20>>>(dev_data,dev_centers,dev_clusters,dev_change_clusters);
        HANDLE_ERROR( cudaMemcpy( clusters, dev_clusters, NUM_RECORDS*sizeof(float), cudaMemcpyDeviceToHost) );
        cout << "Clusters: ";
        for(int i = 0; i < NUM_RECORDS; i++){
            cout << clusters[i] << " ";
        }
        cout << endl;
        HANDLE_ERROR( cudaMemcpy( change_clusters, dev_change_clusters, sizeof(bool), cudaMemcpyDeviceToHost) );

        cout << "Change_clusters is: " << *change_clusters << endl;
        counter--;
    }
    
    HANDLE_ERROR( cudaMalloc((void**)&dev_locations, NUM_RECORDS*sizeof(int)) );
    HANDLE_ERROR( cudaMemcpy(dev_locations, locations, NUM_RECORDS*sizeof(int), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMalloc((void**)&dev_mapping, NUM_CLUSTERS*sizeof(int)));
    HANDLE_ERROR( cudaMemset(dev_mapping, 0, sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**)&dev_max_loc,sizeof(int)));
    HANDLE_ERROR( cudaMemcpy(dev_max_loc, max_loc, sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMalloc((void**)&dev_index, sizeof(int)) );
    HANDLE_ERROR( cudaMemset(dev_index, 0, sizeof(int)) );
    
    classify<<<1,20>>>(dev_clusters,dev_locations,dev_mapping,dev_max_loc,dev_index);
    int* index = new int;
    int* mapping = new int[5];
    HANDLE_ERROR( cudaMemcpy(clusters,dev_clusters, NUM_RECORDS*sizeof(int), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(mapping,dev_mapping, NUM_CLUSTERS*sizeof(int), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(index, dev_index, sizeof(int), cudaMemcpyDeviceToHost) );
    
    
    cout << "Final clusters: " << endl;
    for(int i = 0; i < NUM_RECORDS; i++){
        cout << clusters[i] << " ";
    }
    cout << endl;


    // get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    printf( "Time to analyze:  %3.1f ms\n", elapsedTime );


    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );
    cudaFree( dev_data );
    cudaFree( dev_centers );
    cudaFree( dev_clusters );
    cudaFree( dev_change_clusters );
    cudaFree( dev_max_loc );

    
    return 0;
}
