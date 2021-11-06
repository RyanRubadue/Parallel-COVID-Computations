#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace std;

void find_range(float x[], int length, int num_classes, float centroids[]){
    float* min = min_element(x,x+length);
    float* max = max_element(x,x+length);
    float range = *max - *min;
    float group_size = range / num_classes;
    for(int i =0; i < num_classes; i++){
        centroids[i] = *min + group_size * i;
    }
}

float* calculate_centers(float x[], int length, int clusters[], int num_clusters){
    
    float* centroids = new float[num_clusters];
    int* counts = new int[num_clusters];
    
    for(int i = 0; i < num_clusters; i++){
        centroids[i] = 0.0;
        counts[i] = 0;
    }
    
    for(int i = 0; i < length; i++){
        centroids[clusters[i]] += x[i];
        counts[clusters[i]] += 1;
    }
    for(int i = 0; i < num_clusters; i++){
        if(counts[i] != 0){
            centroids[i] = centroids[i] / counts[i];
        }
    }
    
    return centroids;
}

int* compare(float x[], int length, float centroids[], int counter, int results[]) {
    for(int i = 0; i < 5; i++){
        cout << centroids[i] << " ";
    }
    cout << endl;
    bool change_cluster = false;
    for(int i = 0; i < length; i++){
        float min_diff = abs(x[i]-centroids[results[i]]);

        for(int j = 0; j < 5; j++){
            float diff = abs(x[i] - centroids[j]);
            if (diff < min_diff){
                min_diff = diff;
                cout << "i: " << i << " original: " << results[i] << " after: ";
                if (results[i] != j){
                    change_cluster = true;
                }
                results[i] = j;
                cout << results[i] << endl;
            }
            if (abs(min_diff - 0) < 0.0001){
                break;
            }
        }
    }
    cout << "Clusters: " << endl;
    for(int i = 0; i < length; i++){
        cout << results[i] << " ";
    }
    cout << endl;
    if ( (change_cluster == true) && (counter > 0)){
        float* new_centers = calculate_centers(x, length, results,5);
        int* new_results = compare(x, length, new_centers, counter - 1, results);
        return new_results;
    }
    
    return results;
}

void classify(int x[], int locations[], int length){
    
    int* max_loc = max_element(locations,locations+length);
    vector<vector<int>> loc_list;
    
    for(int i = 0; i <= *max_loc; i++){
        vector<int> temp;
        loc_list.push_back(temp);
    }
    
    for(int i = 0; i < length; i++){
        loc_list.at(locations[i]).push_back(i);
    }
    
    for(int i = 0; i <= *max_loc; i++){
        int count[5] = {0};
        for(int j = 0; j < loc_list.at(i).size(); j++){
            count[x[loc_list.at(i).at(j)]]++;
        }
        int max = count[0];
        int index = 0;
        for(int j = 1; j < 5; j++){
            if (count[j] > max){
                max = count[j];
                index = j;
                }
        }
        for(int j = 0; j < loc_list.at(i).size(); j++){
            x[loc_list.at(i).at(j)] = index;
        }
    }
}

int main() {

    float input[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    int length = 20;
    float centroids[5] = {0};
    int num_classes = 5;
    int counter = 20;
    
    find_range(input,20,num_classes,centroids);
    cout << "Initial centroids: " << endl;
    for(int i = 0; i < 5; i++){
        cout << centroids[i] <<  " ";
    }
    cout << endl;
    int* result = new int[length] ();
    compare(input, length, centroids, counter, result);
    cout << "Results: " << endl;
    for(int i = 0; i < length; i++){
        cout << result[i] << " ";
    }
    cout << endl;
    int* locations = new int[length];
    for(int i = 0; i < length; i++){
        locations[i] = rand() % 10;
        cout << locations[i] << " ";
    }
    cout << endl;
    classify(result,locations,length);
    for(int i = 0; i < length; i++){
        cout << result[i] << " ";
    }
    cout << endl;
    
    return 0;
}
