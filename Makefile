# Version to compile
TARGET= GV_main_khuramIdea

# Compile and link flags
CC	= g++
CFLAGS	= `pkg-config opencv --cflags`  

# Compilation (add flags as needed)
CPPFLAGS= -Wall -g -std=c++11 -Wno-deprecated -g -O0 -fno-inline

# Linking (Add flags as needed)
LDFLAGS	+= `pkg-config opencv --libs` -lBLAS -lLIBLINEAR -lcblas -latlas -lcudart -lcublas -lcurand -lpthread -lglog -lprotobuf -lleveldb -lsnappy -lboost_system -lhdf5_hl -lhdf5

ARCH 	= -arch x86_64

SRC	= $(TARGET).cpp  kyheader.cpp CmFile.cpp CmShow.cpp DataSetVOC.cpp FilterTIG.cpp Objectness.cpp
OBJ	= $(SRC:.cpp=.o)
OBJ2	= daxpy.c.o ddot.c.o dnrm2.c.o dscal.c.o

# Should be -Ipathtodir/
#INCLUDES=/home/gonzalo/Downloads/bing/BingObjectnessCVPR14/Objectness-master/Src/LibLinear/
#INCLUDES=/home/gonzalo/Downloads/bing/BingObjectnessCVPR14/Objectness-master/Src/
#LIBS = /home/gonzalo/Downloads/bing/BingObjectnessCVPR14/Objectness-master/Src/


INCLUDES=-I/home/gonzalo/Downloads/bing/BingObjectnessCVPR14/Objectness-master/Src/ -I/usr/local/include -I/home/gonzalo/Downloads/caffe/caffe-master/build/src -I/home/gonzalo/Downloads/caffe/caffe-master/include -I/usr/local/cuda-6.0/include

LIBS =libcaffe.a -L/home/gonzalo/Downloads/bing/BingObjectnessCVPR14/Objectness-master/Src/ -L/usr/lib/x86_64-linux-gnu/ -L/usr/local/lib -L/usr/lib -L/home/gonzalo/Downloads/x264-snapshot-20131020-2245-stable/ -L/usr/local/cuda-6.0/lib64 -L/usr/local/cuda-6.0/lib


	
default: $(TARGET)

$(TARGET): $(OBJ)
	$(CC)  -o $@ $(OBJ) $(OBJ2) $(LIBS) $(LDFLAGS)

%.o: %.cpp
	$(CC)  $(CPPFLAGS) $(CFLAGS) $(INCLUDES) -c $< -o $@

Debug: $(TARGET)	

clean:
	rm  $(TARGET) $(OBJ)

.deps: $(SRC)
	$(CC) -o .deps $(CPPFLAGS) $(INCLUDES) -M -E $(SRC)

#include dependencies
#include .deps 


