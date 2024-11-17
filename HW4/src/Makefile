# Makefile

CXX = mpicxx
CXXFLAGS = -O3 -std=c++11
TARGET = matmul
OBJECTS = main.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

main.o: main.cc
	$(CXX) $(CXXFLAGS) -c main.cc

clean:
	rm -f $(TARGET) $(OBJECTS)