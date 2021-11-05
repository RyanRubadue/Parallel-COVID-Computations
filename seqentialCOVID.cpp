#include <iostream>
#include <vector>
#include <cstdio>
#include <ctime>
#include "OWID.h"

using namespace std;

const string file_name = "Data/owid-covid-data-2021-11-03.csv";

vector<OWID> read_data_from_csv (vector<OWID> OWIDs)
{
	ifstream file;
	file.open(file_name);
	
	if (!file.is_open())
    {
        cout << "Path Wrong!" << endl;
        exit(EXIT_FAILURE);
    }
	
	int lines = -1;
	string line;
	while (getline(file, line)) 
	{
		lines++;
		if (lines == 0)
			continue;

		OWID o(line);
		OWIDs.push_back(o);
	}
	cout << "Records read: " << lines << endl;
	return OWIDs;
}

int main()
{
	clock_t start;
	start = clock();

	// Read file
	vector<OWID> OWIDs;
	OWIDs = read_data_from_csv(OWIDs);
	
	cout << "Read Data from CSV: " << ( clock() - start ) << " ms" << endl;
}