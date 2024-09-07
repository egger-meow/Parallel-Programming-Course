#include <bits/stdc++.h>
using namespace std;

int main()
{
    srand(time(NULL));

    unsigned long long numToss = 938943248;
    unsigned long long numInCircle = 0;

    for (unsigned long long i = 0; i < numToss; i++) {
        float tossX = ((float) rand() / RAND_MAX) * 2 - 1 ;
        float tossY = ((float) rand() / RAND_MAX) * 2 - 1 ;

        float disToCenter = tossX * tossX + tossY * tossY;
        if (disToCenter <= 1)
            numInCircle ++;
    }
    double estimatedPI = 4 * (double) numInCircle / (double) numToss;

    cout << estimatedPI << endl;
}