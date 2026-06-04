TOOLCHAIN_DIR := C:/Users/wyc18/Downloads/loongson-gnu-toolchain-8.3-i686-mingw-loongarch64-linux-gnu-rc1.6/loongson-gnu-toolchain-8.3-i686-mingw-loongarch64-linux-gnu-rc1.6
CXX := $(TOOLCHAIN_DIR)/bin/loongarch64-linux-gnu-g++.exe
SYSROOT := ./sysroot

CXXFLAGS := -O2 -std=c++17 \
    -I$(SYSROOT)/include/opencv4

LDFLAGS := -L$(SYSROOT)/lib \
    -lopencv_dnn -lopencv_imgproc -lopencv_imgcodecs \
    -lopencv_videoio -lopencv_highgui -lopencv_core \
    -lpthread -ldl -lm

TARGET := runhorse_recognize
TARGET_HTTP := runhorse_http

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

http: $(TARGET_HTTP)

$(TARGET_HTTP): main_http.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET) $(TARGET_HTTP)

.PHONY: all http clean
