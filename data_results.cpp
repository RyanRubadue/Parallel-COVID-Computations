#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <limits.h>

#include "gnuplot-iostream.h"
using namespace std;

float correlation_coefficient(double xValues[], double yValues[], int len1) {

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

void linear_regression(double xValues[], double yValues[], int len1, double &a, double &b)
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

    a = ((sumy * sumxSquared) - (sumx * sumxy)) / ((len1 * sumxSquared) - pow(sumx, 2));
    b = ((len1 * sumxy) - (sumx * sumy)) / ((len1 * sumxSquared) - pow(sumx, 2));
    int ret[] = { a, b };
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


// Demo function for verifying that gnuplot is functional
void demoGnuPlot() {
    Gnuplot gp("\"C:\\Program Files\\code\\gnuplot\\bin\\gnuplot.exe\""); //must have the exact path to the gnuplot binary on local machine
    random_device rd;
    mt19937 mt(rd());
    normal_distribution<double> normdist(0., 1);

    vector<double> v0, v1;
    for (int i = 0; i < 1000; i++) {
        v0.push_back(normdist(mt));
        v1.push_back(normdist(mt));
    }
    partial_sum(v0.begin(), v0.end(), v0.begin());
    partial_sum(v1.begin(), v1.end(), v1.begin());

    gp << "set title 'Graph of two random lines'\n"; //must end gnuplot commands in newline character or will fail
    gp << "plot '-' with lines title 'v0'," 
        << "'-' with lines title 'v1'\n"; //dashes tell gnuplot to listen on stdin
    gp.send(v0);
    gp.send(v1);

    cin.get();
}


void plot_best_fit(double attributeA[], double attributeB[], string nameA, string nameB, int len, double a, double b) {
    Gnuplot gp("\"C:\\Program Files\\code\\gnuplot\\bin\\gnuplot.exe\"");
    vector<double> scatter_data;
    vector<double> best_fit_bounds;
    vector<double> temp;
    int maxA = INT_MIN;
    int minA = INT_MAX;

    for (int i = 0; i < len; i++) {
        temp = { attributeA[i], attributeB[i] };
        scatter_data.insert(scatter_data.end(), temp.begin(), temp.end());

        maxA = max(maxA, attributeA[i]);
        minA = min(minA, attributeA[i]);
    }

    //Store two points for best fit line on min/max x values displayed
    temp = { double(minA), minA * a + b };
    best_fit_bounds.insert(best_fit_bounds.end(), temp.begin(), temp.end());
    temp = { double(maxA), maxA * a + b };
    best_fit_bounds.insert(best_fit_bounds.end(), temp.begin(), temp.end());

    gp << "set title 'Graph of' << stringA <<  'lines'\n";
    gp << "plot '-' with points title 'Scatter Points',"
        << "'-' with lines title 'Best Fit Line'\n";
    
    gp.send(scatter_data);
    gp.send(best_fit_bounds);
    cin.get();
}

int main()
{
    // test gnuplot on random data
    //demoGnuPlot();
    std::cout << "Beginning Display of Analyzed Data Results\n\n";

    // Mock attribute names
    string att_a = "Vaccinations / Million";
    string att_b = "Percent of Country Generally Open"; // Expect Postive Correlation with A
    string att_c = "Deaths / Million"; // Expect Negative Correlation with A
    string att_d = "Commute Time"; // No correlation with A

    // Change explicit array size declaration
    // Definition of mock input arrays
    const int data_len = 4;
    double data[data_len][6] = { { 1, 2, 3 , 4, 5, 6}, { 5, 0, 2, 4, 8, 12}, { 20, 15, 11, 5, 7, 0 }, { 22, 21, 22, 21, 22, 21 }};
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
                double a, b;
                linear_regression(data[i], data[j], len1, a, b);
                cout << "a" << a << "b" << b;

                plot_best_fit(data[i], data[j], attributes[i], attributes[j], len1, a, b);
                return 0;
            }


        }
    }

    //Display the most strongly positive/negative correlations
    sort(begin(correlationValues), end(correlationValues));
    print_strongest_correlation(correlationValues, sizeof(correlationValues) / sizeof(correlationValues[0]));
}

