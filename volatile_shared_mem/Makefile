HIPCC_FLAGS = -g -O2 -Rpass-analysis=kernel-resource-usage -save-temps -std=c++17 -Wno-unused-result -Wno-unused-function -Wno-gnu-line-marker -Wno-pass-failed -Wno-c++17-extensions -x hip --offload-arch=gfx90a

PELEC_FLAGS = -DUSE_VOLATILE

SRC = reverse.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = reverse
DEP = $(OBJ:.o=.d)

# Build target
all: $(TARGET)

# Compile source files
%.o: %.cpp
	hipcc $(HIPCC_FLAGS) $(PELEC_FLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	hipcc -o $@ $^

-include $(DEP)

# Clean target
clean:
	rm -f $(OBJ) $(TARGET) $(DEP)
