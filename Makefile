PROJECT := torch_monitor
CONFIGS := Makefile.config

include $(CONFIGS)

.PHONY: clean all objects install

CC := g++

LIB_DIR := lib/
INC_DIR := include/
SRC_DIR := src/
BUILD_DIR := build/
CUR_DIR = $(shell pwd)/

LIB := $(LIB_DIR)lib$(PROJECT).so

ifdef DEBUG
OFLAGS += -g -DDEBUG
else
OFLAGS += -g -O3
endif

CFLAGS := -fPIC -std=c++17 $(OFLAGS) -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include -I$(PYTHON_INCLUDE_DIR)
LDFLAGS := -fPIC -shared -L$(TORCH_DIR)/lib -Wl,-rpath=$(TORCH_DIR)/lib -L$(PYTHON_LIB_DIR) -Wl,-rpath=$(PYTHON_LIB_DIR)
# XXX(Keren): Werid problems on travis if libraries are at the end of LDFLAGS
LIBRARIES := -lc10 -ltorch -ltorch_cpu -lpython3

SRCS := $(shell find $(SRC_DIR) -maxdepth 3 -name "*.cc")
OBJECTS := $(addprefix $(BUILD_DIR), $(patsubst %.cc, %.o, $(SRCS)))
OBJECTS_DIR := $(sort $(addprefix $(BUILD_DIR), $(dir $(SRCS))))

all: dirs objects lib

ifdef PREFIX
install: all
endif

dirs: $(OBJECTS_DIR) $(LIB_DIR)
objects: $(OBJECTS)
lib: $(LIB)

$(OBJECTS_DIR):
	mkdir -p $@

$(LIB_DIR):
	mkdir -p $@

$(LIB): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBRARIES)

$(OBJECTS): $(BUILD_DIR)%.o : %.cc
	$(CC) $(CFLAGS) -I$(INC_DIR) -o $@ -c $<

clean:
	-rm -rf $(BUILD_DIR) $(LIB_DIR)

ifdef PREFIX
# Do not install main binary
install:
	mkdir -p $(PREFIX)/$(LIB_DIR)
	mkdir -p $(PREFIX)/$(INC_DIR)
	cp -rf $(LIB_DIR) $(PREFIX)
	cp -rf $(INC_DIR)$(PROJECT).h $(PREFIX)/$(INC_DIR)
endif

#utils
print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true
