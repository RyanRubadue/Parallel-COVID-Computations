#ifndef __OW_ID__
#define __OW_ID__
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct owid
{
	char *iso_code;
	char *continent;
	char *location;
	char *date;
	int total_cases;
	int new_cases;
	int total_deaths;
	int new_deaths;
	float total_cases_per_million;
	float new_cases_per_million;
	float total_deaths_per_million;
	float new_deaths_per_million;
	int icu_patients;
	float icu_patients_per_million;
	int hosp_patients;
	float hosp_patients_per_million;
	int new_tests;
	int total_tests;
	float positive_rate;
	float tests_per_case;
	int total_vaccinations;
	int people_vaccinated;
	int people_fully_vaccinated;
	int total_boosters;
	int new_vaccinations;
	int population;
	float median_age;
};

owid initialize_values(struct owid o, char *input_string)
{
	char *field;
	int field_count = 0;
	while ((field = strsep(&input_string, ",")) != NULL)
	{
		if (field_count == 0) o.iso_code = strdup(field);
		if (field_count == 1) o.continent = strdup(field);
		if (field_count == 2) o.location = strdup(field);
		if (field_count == 3) o.date = strdup(field);
		if (field_count == 4) o.total_cases = atoi(field);
		if (field_count == 5) o.new_cases = atoi(field);
		if (field_count == 7) o.total_deaths = atoi(field);
		if (field_count == 8) o.new_deaths = atoi(field);
		if (field_count == 10) o.total_cases_per_million = atof(field);
		if (field_count == 11) o.new_cases_per_million = atof(field);
		if (field_count == 13) o.total_deaths_per_million = atof(field);
		if (field_count == 14) o.new_deaths_per_million = atof(field);
		if (field_count == 17) o.icu_patients = atoi(field);
		if (field_count == 18) o.icu_patients_per_million = atof(field);
		if (field_count == 19) o.hosp_patients = atoi(field);
		if (field_count == 20) o.hosp_patients_per_million = atof(field);
		if (field_count == 25) o.new_tests = atoi(field);
		if (field_count == 26) o.total_tests = atoi(field);
		if (field_count == 31) o.positive_rate = atof(field);
		if (field_count == 32) o.tests_per_case = atof(field);
		if (field_count == 34) o.total_vaccinations = atoi(field);
		if (field_count == 35) o.people_vaccinated = atoi(field);
		if (field_count == 36) o.people_fully_vaccinated = atoi(field);
		if (field_count == 37) o.total_boosters = atoi(field);
		if (field_count == 38) o.new_vaccinations = atoi(field);
		if (field_count == 46) o.population = atoi(field);
		if (field_count == 48) o.median_age = atof(field);
		
		field_count++;
	}
	return o;
}

void print_owid(struct owid o)
{
	printf("iso_code: %s\ncontinent: %s\nlocation: %s\ndate: %s\ntotal_cases: %i\nnew_cases: %i\ntotal_deaths: %i\nnew_deaths: %i\ntotal_cases_per_million: %f\nnew_cases_per_million: %f\ntotal_deaths_per_million: %f\nnew_deaths_per_million: %f\nicu_patients: %i\nicu_patients_per_million: %f\nhosp_patients: %i\nhosp_patients_per_million: %f\nnew_tests: %i\ntotal_tests: %i\npositive_rate: %f\ntests_per_case: %f\ntotal_vaccinations: %i\npeople_vaccinated: %i\npeople_fully_vaccinated: %i\ntotal_boosters: %i\nnew_vaccinations: %i\npopulation: %i\nmedian_age: %f\n\n",o.iso_code,o.continent,o.location,o.date,o.total_cases,o.new_cases,o.total_deaths,o.new_deaths,o.total_cases_per_million,o.new_cases_per_million,o.total_deaths_per_million,o.new_deaths_per_million,o.icu_patients,o.icu_patients_per_million,o.hosp_patients,o.hosp_patients_per_million,o.new_tests,o.total_tests,o.positive_rate,o.tests_per_case,o.total_vaccinations,o.people_vaccinated,o.people_fully_vaccinated,o.total_boosters,o.new_vaccinations,o.population,o.median_age);
}

# endif /*__OW_ID__ */

