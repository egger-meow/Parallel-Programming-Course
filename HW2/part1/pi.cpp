#include <bits/stdc++.h>
#include<pthread.h>
#include<ctime>
#include <cstdlib>
using namespace std;

typedef unsigned long long ull;

struct threaData {
    ull tosses;
    ull inCircle;
};



void *threadCalculation(void* arg) 
{
    threaData *data = (threaData*) arg;
    ull inCircle = 0;
    unsigned int seed = time(NULL) ^ pthread_self();

    for (ull i = 0; i < data -> tosses; i++) {
        double tossX = (double)rand_r(&seed) / RAND_MAX;
        double tossY = (double)rand_r(&seed) / RAND_MAX;
        double distance = tossX * tossX + tossY * tossY;
        inCircle += distance <= 1.0 ? 1 : 0;
    }

    data -> inCircle = inCircle;
    return NULL;
}

int main(int argc, char* argv[])
{
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <number_of_threads> <number_of_tosses>" << endl;
        return 1;
    }

    int numThreads = atoi(argv[1]);              
    unsigned long long numToss = atoll(argv[2]); 

    unsigned long long numTossThread = numToss / numThreads;
    unsigned long long numInCircle = 0;

    pthread_t threads[numThreads]; 
    threaData datas[numThreads];

    for (int i = 0; i < numThreads; i++) {
        datas[i] = {numTossThread, 0};
        pthread_create(threads + i, NULL, threadCalculation, datas + i);
    }

    for (int i = 0; i < numThreads; i++) {
        pthread_join(threads[i], NULL);
        numInCircle += datas[i].inCircle;
    }

    double estimatedPI = 4 * (double) numInCircle / (double) numToss;

     printf("%.6f\n", estimatedPI);
}