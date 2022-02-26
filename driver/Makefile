PROJECT := driver
CONFIGS := Makefile.config

include $(CONFIGS)

.PHONY: clean all 

CC := g++

LIB := $(PROJECT).so

CFLAGS := -fPIC -std=c++17 -I$(TORCH_MONITOR_DIR)/include
LDFLAGS := -fPIC -shared -L$(TORCH_MONITOR_DIR)/lib -ltorch_monitor -Wl,-rpath=$(TORCH_MONITOR_DIR)/lib

SRCS := $(PROJECT).cc
OBJECTS := $(PROJECT).o

all: lib

lib: $(LIB)

$(LIB): $(SRCS)
	$(CC) $(CFLAGS) $(LDFLAGS) -I$(TORCH_MONITOR_DIR)/include -o $@ $<

clean:
	-rm -rf $(PROJECT).so

#utils
print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true
