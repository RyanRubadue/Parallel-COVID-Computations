// COVID_DATA.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <cmath>;

using namespace std;



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
        cout << attributeA << " and " << attributeB << " have a strong " << sign << " correlation\n\n";
    }
    else if (abs(correlation) > 0.5) {
        cout << attributeA << " and " << attributeB << " have a moderate " << sign << " correlation\n\n";
    }
    else if (abs(correlation) > 0.3) {
        cout << attributeA << " and " << attributeB << " have a weak " << sign << " correlation\n\n";
    }
    else {
        cout << attributeA << " and " << attributeB << " have little to no correlation\n\n";
        possible_correlation = 0;
    }
    return possible_correlation;
}

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

int linear_regression(double xValues[], double yValues[], int len1)
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


int main()
{
    std::cout << "Beginning Display of Analyzed Data Results\n\n";
    // Definition of mock input arrays
    double a[] = { 1, 2, 3 , 4, 5, 6};
    double b[] = { 5, 0, 2, 4, 8, 12};
    double c[] = { 20, 15, 11, 5, 7, 0 };
    double d[] = { 22, 25, 24, 25, 23, 26 };

    string att_a = "Vaccinations / Million";
    string att_b = "Percent of Country Generally Open"; // Expect Postive Correlation with A
    string att_c = "Deaths / Million"; // Expect Negative Correlation with A
    string att_d = "Commute Time"; // No correlation with A


    int len1 = sizeof(a) / sizeof(a[0]);
    int len2 = sizeof(b) / sizeof(b[0]);

    // Re-work where this check happens
    if (len1 != len2) {
        cout << "Error calculating statistics on arrays. Different array lengths.\n";
        return 1;
    }

    // Determine how strong the correlation between two datasets is
    float correlation_ab = correlation_coefficient(a, b, len1);

    // Print correlation result
    bool possible_correlation = print_correlation_strength(att_a, att_b, correlation_ab);

    // If the datasets are found to be correlated, find best-fit line and plot
    if (possible_correlation) {
        linear_regression(a, b, len1);
    }
}

