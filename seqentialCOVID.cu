#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gputimer.h"
#include "owid.h"

const char * file_name = "Data/owid-covid-data-2021-11-03.csv";
const int line_length = 1024;

int main(int argc, char **argv)
{
	GpuTimer timer;
	timer.Start();
	
	FILE * fp = fopen(file_name, "r");
	char * file_content = (char*)malloc(line_length);
	int lines = -1;
	while (fgets(file_content, line_length, fp)) {
		lines++;
	}
	rewind(fp);
	
	struct owid data[lines];
	int i = 0;
	while (fgets(file_content, line_length, fp)) {
		if (i != 0){
			data[i-1] = initialize_values(data[i-1],file_content);
		}
		i++;
	}
	
	timer.Stop();
	printf("Records read: %i\n", lines);
	printf("read_data_from_csv: %g ms.\n", timer.Elapsed());
}