#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <chrono>

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
    int startRow, int endRow,
    int startRCol, int endCol,
    int maxIterations,
    int output[])
{
  float dx = (x1 - x0) / width;
  float dy = (y1 - y0) / height;

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
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    int blocksX = args -> numThreads + 1;
    int blocksY = args -> numThreads;
    int rowsPerThread = args -> height / blocksY;
    int colsPerThread = args -> width / blocksX;

    for (int i = 0; i < blocksY; i++) {
        for (int j = 0; j < blocksX; j++) {
            int id = i * blocksX + j;
            if ((id - args -> threadId) % args -> numThreads == 0) {
                int rowStart = i * rowsPerThread;
                int rowEnd = i == blocksY - 1 ? args -> height : rowStart + rowsPerThread;
                int colStart = j * colsPerThread;
                int colEnd = j == blocksX - 1  ? args -> width : colStart + colsPerThread;
                mandelbrotSerial2(
                    args -> x0, args -> y0, args -> x1, args ->  y1,
                    args -> width, args -> height,
                    rowStart, rowEnd,
                    colStart, colEnd,
                    args -> maxIterations,
                    args -> output
                );
            } 
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    printf("Thread %d completed in %.6f seconds\n", args->threadId, elapsed.count());
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
