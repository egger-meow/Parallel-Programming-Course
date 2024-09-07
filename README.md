# Docker Setup for PP2024Fall

## Overview
This repository contains the setup for the **PP2024Fall** Docker environment. It includes the necessary steps to build and run the Docker container for your project.

## Build the Docker Image
To build the Docker image, use the following command:
```bash
docker build -t pp2024fall .
```
To run it:
```bash
docker run --name pp -v /c/Users/Owner/2024Fall/PP:/home/jjmow -it pp2024fall
```
