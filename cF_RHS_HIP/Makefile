HIPCC_FLAGS = -Rpass-analysis=kernel-resource-usage -save-temps -O2 -g -DNDEBUG -std=c++17 -Wall -Wextra -pedantic -Wno-unused-function -Wno-gnu-line-marker -faligned-new -Wunreachable-code -Wnull-dereference -Wfloat-conversion -Wshadow -Woverloaded-virtual -Wno-pass-failed -Wno-c++17-extensions -ffp-exception-behavior=maytrap -x hip --offload-arch=gfx90a -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -m64 -munsafe-fp-atomics -pthread -MD

SRC = ReactorCvode.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = cF_RHS_HIP
DEP = $(OBJ:.o=.d)

# Build target
all: $(TARGET)

# Compile source files
%.o: %.cpp
	hipcc $(HIPCC_FLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	hipcc -o $@ $^

-include $(DEP)

# Clean target
clean:
	rm -f $(OBJ) $(TARGET)
