#ifndef __OW_ID__
#define __OW_ID__

#include <string>
#include <bits/stdc++.h>
using namespace std;

class OWID
{
	public:

		OWID(string line)
		{
			string field;
			double field_count = 0;
			stringstream s (line);
			while (getline(s, field, ','))
			{
				if (field_count == 0) iso_code = field;
				if (field_count == 1) continent = field;
				if (field_count == 2) location = field;
				if (field_count == 3) date = field;
				if (field_count == 4) total_cases = field.empty() ? 0 : stod(field);
				if (field_count == 5) new_cases = field.empty() ? 0 : stod(field);
				if (field_count == 7) total_deaths = field.empty() ? 0 : stod(field);
				if (field_count == 8) new_deaths = field.empty() ? 0 : stod(field);
				if (field_count == 10) total_cases_per_million = field.empty() ? 0 : stod(field);
				if (field_count == 11) new_cases_per_million = field.empty() ? 0 : stod(field);
				if (field_count == 13) total_deaths_per_million = field.empty() ? 0 : stod(field);
				if (field_count == 14) new_deaths_per_million = field.empty() ? 0 : stod(field);
				if (field_count == 17) icu_patients = field.empty() ? 0 : stod(field);
				if (field_count == 18) icu_patients_per_million = field.empty() ? 0 : stod(field);
				if (field_count == 19) hosp_patients = field.empty() ? 0 : stod(field);
				if (field_count == 20) hosp_patients_per_million = field.empty() ? 0 : stod(field);
				if (field_count == 25) new_tests = field.empty() ? 0 : stod(field);
				if (field_count == 26) total_tests = field.empty() ? 0 : stod(field);
				if (field_count == 31) positive_rate = field.empty() ? 0 : stod(field);
				if (field_count == 32) tests_per_case = field.empty() ? 0 : stod(field);
				if (field_count == 34) total_vaccinations = field.empty() ? 0 : stod(field);
				if (field_count == 35) people_vaccinated = field.empty() ? 0 : stod(field);
				if (field_count == 36) people_fully_vaccinated = field.empty() ? 0 : stod(field);
				if (field_count == 37) total_boosters = field.empty() ? 0 : stod(field);
				if (field_count == 38) new_vaccinations = field.empty() ? 0 : stod(field);
				if (field_count == 48) median_age = field.empty() ? 0 : stod(field);
				
				field_count++;
			}
		}

		string iso_code;
		string continent;
		string location;
		string date;
		double total_cases;
		double new_cases;
		double total_deaths;
		double new_deaths;
		double total_cases_per_million;
		double new_cases_per_million;
		double total_deaths_per_million;
		double new_deaths_per_million;
		double icu_patients;
		double icu_patients_per_million;
		double hosp_patients;
		double hosp_patients_per_million;
		double new_tests;
		double total_tests;
		double positive_rate;
		double tests_per_case;
		double total_vaccinations;
		double people_vaccinated;
		double people_fully_vaccinated;
		double total_boosters;
		double new_vaccinations;
		double median_age;
};

# endif /*__OW_ID__ */