#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <chrono>

static float x0, x1;
static float y0, y1;
static unsigned int width, height;
static int maxIterations;
static int *output;
static int numThreads;

static int blocksX;
static int blocksY;
static int rowsPerThread;
static int colsPerThread;
// extern void mandelbrotSerial(
//     float x0, float y0, float x1, float y1,
//     int width, int height,
//     int startRow, int numRows,
//     int maxIterations,
//     int output[]);

inline int mandel(float c_re, float c_im, int count)
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
    int startRow, int endRow,
    int startRCol, int endCol)
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
static void workerThreadStart(int *const threadId)
{
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    int blocksX = args -> numThreads + 1;
    int blocksY = args -> numThreads;
    int rowsPerThread = args -> height / blocksY;
    int colsPerThread = args -> width / blocksX;

    for (int i = 0; i < blocksY; i++) {
        for (int j = 0; j < blocksX; j++) {
            if (tid == 0) {
                int rowStart = i * rowsPerThread;
                int rowEnd = i == blocksY - 1 ? height : rowStart + rowsPerThread;
                int colStart = j * colsPerThread;
                int colEnd = j == blocksX - 1  ? width : colStart + colsPerThread;
                mandelbrotSerial2(
                    rowStart, rowEnd,
                    colStart, colEnd
                );
            } 
            tid = (tid + 1) % numThreads;
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
    int _numThreads,
    float _x0, float _y0, float _x1, float _y1,
    int _width, int _height,
    int _maxIterations, int _output[])
{
    static constexpr int MAX_THREADS = 32;
    
    x0 = _x0;
    x1 = _x1;
    y0 = _y0;
    y1 = _y1;
    width = _width;
    height = _height;
    maxIterations = _maxIterations;
    output = _output;
    numThreads = _numThreads;
    float ratio = static_cast<float>(width) / static_cast<float>(height) * numThreads + 0.5;

    blocksX = static_cast<int>(ratio) + 1;
    blocksY = numThreads + 1;
    rowsPerThread = height / blocksY;
    colsPerThread = width / blocksX;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    static std::thread workers[MAX_THREADS];
    static int ids[MAX_THREADS] = {};


    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < numThreads; i++)
    {
        ids[i-1] = i;
        workers[i] = std::thread(workerThreadStart, &ids[i]);
    }

    workerThreadStart(&ids[0]);

    // join worker threads
    for (int i = 1; i < numThreads; i++)
    {
        workers[i].join();
    }
}
