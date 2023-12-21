#include <iostream>
#include <math.h>

class LinearRegressionModel{
public:
    // Hypothesis function.
    float GetHypothesis(int n, float* features, float* theta)
    {
        float result = theta[0];
        for(int i=0; i<n; i++)
        {
            result+=features[i]*theta[i+1];
        }
        return result;
    }

    // Cost function.
    float GetMeanSquaredError(int n, float* yActualValue, float* yPredictedValue)
    {
        float result = 0.0f;
        for(int i=0; i<n; i++)
        {
            result += pow(yActualValue[i] - yPredictedValue[i], 2);
        }
        return result/n;
    }

    // Gradient Descent function.
};

int main()
{
    int n = 5;
    float yActualValue[] = {
        1, 2, 3, 4, 5
    };
    float yPredictedValue[] = {
        0.5f, 1.5f, 2.8f, 4.5f, 10.0f
    };

    float yPredictedValueTheBest[] = {
        1.1f, 2, 3.1f, 4, 5.1f
    };

    LinearRegressionModel linearRegressionModel{};

    float result = linearRegressionModel.GetMeanSquaredError(n, yActualValue, yPredictedValue);

    std::cout<<"1. Mean squared error: "<<result<<"\n";

    result = linearRegressionModel.GetMeanSquaredError(n, yActualValue, yPredictedValueTheBest);

    std::cout<<"2. Mean squared error: "<<result<<"\n";

    float xFeatures[]={
        1, 2, 3
    };
    float theta[]={
        0.5f, 0.3f, 0.4f
    };

    float hypothesisResult = linearRegressionModel.GetHypothesis(
        n, xFeatures, theta
    );

    std::cout<<"3. Hypothesis result: "<<hypothesisResult<<"\n";
}