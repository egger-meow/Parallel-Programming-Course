Parallel Programming 2024Fall

## Overview
This repo includes the content of my dedication to the course and the necessary steps to build and run the Docker container for your project.

## Build the Docker Image
To build the Docker image:
```bash
docker build -t pp2024fall .
```
To run it:
```bash
docker run --name pp -v /c/Users/Owner/2024Fall/PP:/home/jjmow -it pp2024fall
# docker run --name pp -v /d/course/2024Fall/Parallel-Programming-Course:/home/jjmow -it pp2024fall
```