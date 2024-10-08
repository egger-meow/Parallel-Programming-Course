#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[])
{
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <number_of_threads> <number_of_tosses>" << endl;
        return 1;
    }

    int numThreads = atoi(argv[1]);              
    unsigned long long numToss = atoll(argv[2]); 



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