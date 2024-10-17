
CXX = g++

CXXFLAGS = -Wall -g -pthread # Enable warnings, debugging, and link pthread

TARGET = pi.out

.PHONY: force clean

$(TARGET): pi.cpp | force
	@$(CXX) $(CXXFLAGS) -o $(TARGET) pi.cpp

force:

# Clean up build files
clean:
	@rm -f $(TARGET)
	@echo "Cleaned up $(TARGET)."
