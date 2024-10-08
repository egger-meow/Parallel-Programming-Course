# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -g -pthread # Link with pthread

# Target executable
TARGET = pi.out

# Build the target directly from pi.cpp
$(TARGET): pi.cpp
	@$(CXX) $(CXXFLAGS) -o $(TARGET) pi.cpp

# Clean up build files
clean:
	@rm -f $(TARGET)