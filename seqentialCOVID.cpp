#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <vector>
#include <bits/stdc++.h>

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

const int column_number = 4; // Column #
const int length = 130600;
const string file_name = "Data/owid-covid-data-2021-11-03.csv"; // File location

void build_input(float input[], int location[]){
	ifstream file;
	int lines = -1, loc = 0;
	string line, cur_loc, field;
	file.open(file_name);
	while (getline(file, line)) 
	{
		lines++;
		if (lines == 0)
			continue;
		double field_count = 0;
		stringstream s (line);
		while (getline(s, field, ','))
		{
			if (field_count == 2 && cur_loc != field){
				loc++;
				cur_loc = field;
				location[lines - 1] = loc;
				//cout << "Location #: " << loc << "\t| Location: " << cur_loc << endl;
			}
			else if (field_count == 2 && cur_loc == field)
			{
				location[lines - 1] = loc;
			}
			if (field_count == column_number) 
			{
				input[lines - 1] = field.empty() ? 0 : stof(field);
			}
			field_count++;
		}
	}
}

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
	//for(int i = 0; i < 5; i++){
	//	cout << centroids[i] << " ";
	//}
	//cout << endl;
	bool change_cluster = false;
	for(int i = 0; i < length; i++){
		float min_diff = abs(x[i]-centroids[results[i]]);

		for(int j = 0; j < 5; j++){
			float diff = abs(x[i] - centroids[j]);
			if (diff < min_diff){
				min_diff = diff;
				//cout << "i: " << i << " original: " << results[i] << " after: ";
				if (results[i] != j){
					change_cluster = true;
				}
				results[i] = j;
				//cout << results[i] << endl;
			}
			if (abs(min_diff - 0) < 0.0001){
				break;
			}
		}
	}
	//cout << "Clusters: " << endl;
	//for(int i = 0; i < length; i++){
	//	cout << results[i] << " ";
	//}
	//cout << endl;
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
	float* input = new float[length];
	int* locations = new int[length];
	int* result = new int[length] ();
	float centroids[5] = {0};
	int num_classes = 5;
	int counter = 20;

	clock_t start;
	start = clock();

	build_input(input, locations);

	cout << "build_input: " << ( clock() - start ) << " ms" << endl;
	start = clock();
	
	find_range(input,20,num_classes,centroids);
	//cout << "Initial centroids: " << endl;
	//for(int i = 0; i < 5; i++){
	//	cout << centroids[i] <<  " ";
	//}
	//cout << endl;

	cout << "find_range: " << ( clock() - start ) << " ms" << endl;
	start = clock();
	
	compare(input, length, centroids, counter, result);
	//cout << "Results: " << endl;
	//for(int i = 0; i < length; i++){
	//	cout << result[i] << " ";
	//}
	//cout << endl;

	cout << "compare: " << ( clock() - start ) << " ms" << endl;
	start = clock();

	//for(int i = 0; i < length; i++){
	//	locations[i] = rand() % 10;
	//	cout << locations[i] << " ";
	//}
	//cout << endl;

	classify(result,locations,length);
	//for(int i = 0; i < length; i++){
	//	cout << result[i] << " ";
	//}
	//cout << endl;

	cout << "classify: " << ( clock() - start ) << " ms" << endl;
	
	return 0;
}
