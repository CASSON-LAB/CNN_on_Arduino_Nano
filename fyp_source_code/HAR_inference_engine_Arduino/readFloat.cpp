#include "readFloat.h"
#include "Arduino.h"
//read four bytes and combine them by exploiting the property of union
float readingFloat(){
   union {
    byte data[4];
    float value;
  } convertFloat;
  while(Serial.available()<4){
    //do nothing
    }
  size_t bytesRead = Serial.readBytes(convertFloat.data,4);
  return convertFloat.value;
  
}

void cpyArr(float* input, float* data){
  for (int i=0;i<1152;++i){
    input[i]=data[i];
  }
}
