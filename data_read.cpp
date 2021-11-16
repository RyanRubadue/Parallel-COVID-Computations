#include <iostream>
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
const int length = 130600; // Number of Records to read
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
				cout << "Location #: " << loc << "\t| Location: " << cur_loc << endl;
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

int main() {
	cout << "Starting..." << endl;

	float* input_1 = new float[length];
	float* input_2 = new float[length];
	float* input_3 = new float[length];
	float* input_4 = new float[length];
	int* locations = new int[length];

	clock_t start;
	start = clock();

	build_input(input_1, input_2, input_3, input_4, locations);

	cout << "build_input: " << ( clock() - start ) << " ms" << endl;
	
	return 0;
}