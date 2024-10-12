#include <stdio.h>
#include <stdlib.h>
#include <thread>

typedef struct
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
} WorkerArgs;

// extern void mandelbrotSerial(
//     float x0, float y0, float x1, float y1,
//     int width, int height,
//     int startRow, int numRows,
//     int maxIterations,
//     int output[]);

static inline int mandel(float c_re, float c_im, int count)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

//
// MandelbrotSerial --
//
// Compute an image visualizing the mandelbrot set.  The resulting
// array contains the number of iterations required before the complex
// number corresponding to a pixel could be rejected from the set.
//
// * x0, y0, x1, y1 describe the complex coordinates mapping
//   into the image viewport.
// * width, height describe the size of the output image
// * startRow, totalRows describe how much of the image to compute
void mandelbrotSerial2(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int totalRows,
    int startRCol, int totalCols,
    int maxIterations,
    int output[])
{
  float dx = (x1 - x0) / width;
  float dy = (y1 - y0) / height;

  int endRow = startRow + totalRows;
  int endCol = startRCol + totalCols;

  for (int j = startRow; j < endRow; j++)
  {
    for (int i = startRCol; i < endCol; ++i)
    {
      float x = x0 + i * dx;
      float y = y0 + j * dy;

      int index = (j * width + i);
      output[index] = mandel(x, y, maxIterations);
    }
  }
}

//
// workerThreadStart --
//
// Thread entrypoint.
void workerThreadStart(WorkerArgs *const args)
{

    // TODO FOR PP STUDENTS: Implement the body of the worker
    // thread here. Each thread could make a call to mandelbrotSerial()
    // to compute a part of the output image. For example, in a
    // program that uses two threads, thread 0 could compute the top
    // half of the image and thread 1 could compute the bottom half.
    // Of course, you can copy mandelbrotSerial() to this file and
    // modify it to pursue a better performance.
    int rowsPerThread = args -> height / args -> numThreads;
    int colsPerThread = args -> width / (args -> numThreads + 1);
    for (int i = 0; i < args -> numThreads; i++) {
        for (int j = 0; j < args -> numThreads + 1; j++) {
            int id = i * (args -> numThreads + 1) + j;
            if ((id - args -> threadId) % args -> numThreads == 0) {
                int rowStart = i * rowsPerThread;
                int rowEnd = i == args -> numThreads - 1 ? args -> height : rowStart + rowsPerThread;
                int colStart = j * colsPerThread;
                int colEnd = j == args -> numThreads  ? args -> width : colStart + colsPerThread;
                mandelbrotSerial2(
                    args -> x0, args -> y0, args -> x1, args ->  y1,
                    args -> width, args -> height,
                    rowStart, rowEnd - rowStart,
                    colStart, colEnd - colStart,
                    args -> maxIterations,
                    args -> output
                );

            } 
        }
    }
    // int rowStart = args -> threadId * rowsPerThread;
    // int rowEnd = args -> threadId == args -> numThreads - 1 ? args -> height : rowStart + rowsPerThread;
    // mandelbrotSerial2(
    //     args -> x0, args -> y0, args -> x1, args ->  y1,
    //     args -> width, args -> height,
    //     rowStart, rowEnd - rowStart,
    //     args -> maxIterations,
    //     args -> output
    // );

int blockSize = 16;  // Divide the image into 16x16 pixel blocks
int blocksPerThread = (args -> width * args -> height) / (blockSize * blockSize) / args -> numThreads;

for (int block = 0; block < blocksPerThread; block++) {
    int startX = (block % (args -> width / blockSize)) * blockSize;
    int startY = (block / (args -> width / blockSize)) * blockSize;
    mandelbrotSerial2(
        x0, y0, x1, y1, 
        width, height, startY, blockSize, maxIterations, output
        );
}

    // printf("Hello world from thread %d\n", args->threadId);
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    static constexpr int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS] = {};

    for (int i = 0; i < numThreads; i++)
    {
        // TODO FOR PP STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].numThreads = numThreads;
        args[i].output = output;

        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < numThreads; i++)
    {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }

    workerThreadStart(&args[0]);

    // join worker threads
    for (int i = 1; i < numThreads; i++)
    {
        workers[i].join();
    }
}
