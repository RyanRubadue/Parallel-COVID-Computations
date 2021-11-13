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

const int column_number_1 = 4; // Column #
const int column_number_2 = 10; // Column #
const int column_number_3 = 5; // Column #
const int column_number_4 = 35; // Column #
// Mock attribute names
const string att_a = "Total Deaths";
const string att_b = "Deaths Per Million"; 
const string att_c = "New Cases"; 
const string att_d = "People Vaccinated";
const int length = 618; //130600
const string file_name = "Data/owid-covid-data-2021-11-03.csv"; // File location

void build_input(float input_1[], float input_2[], float input_3[], float input_4[], int location[]){

	ifstream file;
	int lines = -1, loc = 0;
	string line, cur_loc, field;
	file.open(file_name);
	while (getline(file, line)) 
	{
		lines++;
		if (lines == 0 || lines-1 >= length)
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

			if (field_count == column_number_1) 
			{
				input_1[lines - 1] = field.empty() ? 0 : stof(field);
			}
			if (field_count == column_number_2) 
			{
				input_2[lines - 1] = field.empty() ? 0 : stof(field);
			}
			if (field_count == column_number_3) 
			{
				input_3[lines - 1] = field.empty() ? 0 : stof(field);
			}
			if (field_count == column_number_4) 
			{
				input_4[lines - 1] = field.empty() ? 0 : stof(field);
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


float correlation_coefficient(int xValues[], int yValues[], int len1) {

	// Calculate mean of each dataset
	float meanx = 0;
	float meany = 0;
	for (int i = 0; i < len1; i++) {
		meanx = meanx + xValues[i];
		meany = meany + yValues[i];
	}
	meanx = meanx / len1;
	meany = meany / len1;

	// Calculate devaition scores and product of deviattion scores
	float ssx = 0;
	float ssy = 0;
	float xy = 0;
	for (int i = 0; i < len1; i++) {
		ssx = ssx + pow(xValues[i] - meanx, 2);
		ssy = ssy + pow(yValues[i] - meany, 2);
		xy = xy + (xValues[i] - meanx) * (yValues[i] - meany);
	}

	// Calculate correlation
	return xy / sqrt(ssx * ssy);
}

int linear_regression(int xValues[], int yValues[], int len1)
{
	double sumx = 0;
	double sumy = 0;
	double sumxy = 0;
	double sumxSquared = 0;
	double sumySquared = 0;

	for (int i = 0; i < len1; i++) {
		sumx = sumx + xValues[i];
		sumy = sumy + yValues[i];
		sumxy = sumxy + xValues[i] * yValues[i];
		sumxSquared = sumxSquared + pow(xValues[i], 2);
		sumySquared = sumySquared + pow(yValues[i], 2);
	}

	double a = ((sumy * sumxSquared) - (sumx * sumxy)) / ((len1 * sumxSquared) - pow(sumx, 2));
	double b = ((len1 * sumxy) - (sumx * sumy)) / ((len1 * sumxSquared) - pow(sumx, 2));
	return 0;
}

bool print_correlation_strength(string attributeA, string attributeB, float correlation) {
	if (abs(correlation) > 1) {
		cout << "Invalid correlation value. Exiting\n";
		return -1;
	}
	bool possible_correlation = 1;
	string sign = "positive";
	if (correlation < 0) {
		sign = "negative";
	}
	if (abs(correlation) > 0.7) {
		cout << attributeA << " and " << attributeB << " have a strong " << sign << " correlation with a value of " << correlation << "\n";
	}
	else if (abs(correlation) > 0.5) {
		cout << attributeA << " and " << attributeB << " have a moderate " << sign << " correlation with a value of " << correlation << "\n";
	}
	else if (abs(correlation) > 0.3) {
		cout << attributeA << " and " << attributeB << " have a weak " << sign << " correlation with a value of " << correlation << "\n";
	}
	else {
		cout << attributeA << " and " << attributeB << " have little to no correlation with a value of " << correlation << "\n";
		possible_correlation = 0;
	}
	return possible_correlation;
}

void print_strongest_correlation(double sorted_correlations[], int len) {
	cout << "\n\nPrinting Strongest Postive and Negative Correlations\n";

	for (int i = 0; i < 3; i++) {
		if (i >= len) return;
		if (sorted_correlations[i] >= 0) break;
		cout << "Number " << i + 1 << " strongest negative correlation: " << sorted_correlations[i] << "\n";
	}

	cout << "\n";

	int positive_print_num = 1;
	for (int i = len -1; i > len - 4; i--) {
		if (i < 0 || sorted_correlations[i] < 0) return;
		// Improve to display the attributes each correlation is associated with 
		cout << "Number " << positive_print_num << " strongest positive correlation: " << sorted_correlations[i] << "\n";
		positive_print_num++;
	}
}

int main() {
	cout << "Starting..." << endl;

	float* input_1 = new float[length];
	float* input_2 = new float[length];
	float* input_3 = new float[length];
	float* input_4 = new float[length];
	int* locations = new int[length];
	int* result_1 = new int[length] ();
	int* result_2 = new int[length] ();
	int* result_3 = new int[length] ();
	int* result_4 = new int[length] ();
	float centroids_1[5] = {0};
	float centroids_2[5] = {0};
	float centroids_3[5] = {0};
	float centroids_4[5] = {0};
	int num_classes = 5;
	int counter = 20;

	clock_t start;
	start = clock();

	build_input(input_1, input_2, input_3, input_4, locations);

	cout << "build_input: " << ( clock() - start ) << " ms" << endl;
	start = clock();
	
	find_range(input_1,20,num_classes,centroids_1);
	find_range(input_2,20,num_classes,centroids_2);
	find_range(input_3,20,num_classes,centroids_3);
	find_range(input_4,20,num_classes,centroids_4);
	//cout << "Initial centroids: " << endl;
	//for(int i = 0; i < 5; i++){
	//	cout << centroids[i] <<  " ";
	//}
	//cout << endl;

	cout << "find_range: " << ( clock() - start ) << " ms" << endl;
	start = clock();
	
	compare(input_1, length, centroids_1, counter, result_1);
	compare(input_2, length, centroids_2, counter, result_2);
	compare(input_3, length, centroids_3, counter, result_3);
	compare(input_4, length, centroids_4, counter, result_4);
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

	classify(result_1,locations,length);
	classify(result_2,locations,length);
	classify(result_3,locations,length);
	classify(result_4,locations,length);
	//for(int i = 0; i < length; i++){
	//	cout << result[i] << " ";
	//}
	//cout << endl;

	cout << "classify: " << ( clock() - start ) << " ms" << endl << endl;
	start = clock();

	cout << "Beginning Display of Analyzed Data Results\n\n";

	// Change explicit array size declaration
	// Definition of mock input arrays
	const int data_len = 4;
	int data[data_len][length] = { *result_1, *result_2, *result_3, *result_4 };
	double correlationValues[(data_len) * (data_len - 1) / 2];
	int correlationCurrIndex = 0;
	string attributes[] = { att_a, att_b, att_c, att_d };

	cout << "Correlation Results:\n\n";
	// Loop through each combination of attributes pairs
	for (int i = 0; i < sizeof(data) / sizeof(data[0]) - 1; i++) {
		for (int j = i + 1; j < sizeof(data) / sizeof(data[0]); j++) {

			int len1 = sizeof(data[i]) / sizeof(data[i][0]);
			int len2 = sizeof(data[j]) / sizeof(data[j][0]);

			if (len1 != len2) {
				cout << "Error calculating statistics on arrays. Different array lengths.\n";
				return 1;
			}

			// Determine how strong the correlation between two datasets is
			float correlation = correlation_coefficient(data[i], data[j], len1);
			if(correlationCurrIndex >= sizeof(correlationValues) / sizeof(correlationValues[0]))
			{
				cout << "Error. Attempted to assign out-of-bounds value in correlation values array.";
				return 1;
			}
			correlationValues[correlationCurrIndex] = correlation;
			correlationCurrIndex++;

			// Print correlation result
			bool possible_correlation = print_correlation_strength(attributes[i], attributes[j], correlation);

			// If the datasets are found to be correlated, find best-fit line and plot
			if (possible_correlation) {
				linear_regression(data[i], data[j], len1);
			}

		}
	}

	//Display the most strongly positive/negative correlations
	sort(begin(correlationValues), end(correlationValues));
	print_strongest_correlation(correlationValues, sizeof(correlationValues) / sizeof(correlationValues[0]));

	cout << "display: " << ( clock() - start ) << " ms" << endl;
	
	return 0;
}
