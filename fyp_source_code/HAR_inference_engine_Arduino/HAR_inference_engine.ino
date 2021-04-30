#include <TensorFlowLite.h>

#include "main_functions.h"
#include "readFloat.h"
#include "har_predictor.h"
#include "har_model_data.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
int input_length;
float arr[1152];
int arrInd=0;
int infTime=0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 100 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  Serial.begin(115200);
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  static tflite::MicroErrorReporter micro_error_reporter;  // NOLINT
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(har_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull all Ops
  
   static tflite::AllOpsResolver micro_op_resolver;  // NOLINT
  

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != 128) ||
      (model_input->dims->data[2] != 9) ||
      (model_input->type != kTfLiteFloat32)) {
    Serial.println("Bad input tensor parameters in model");
    return;
  }

  input_length = model_input->bytes / sizeof(float);


}

void loop() {
  // Attempt to read new data from the accelerometer.
  while(!Serial.available()){
    //do nothing when no input byte
  }
  arr[arrInd++]=readingFloat();
  if(arrInd>=1152){//when the sample window is full, run inference
    arrInd=0;
    cpyArr(model_input->data.f, arr);
    int temp=millis();
    TfLiteStatus invoke_status = interpreter->Invoke();
    temp=millis()-temp;
    infTime+=temp;
    if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed on index");
    return;
    }
    
  // Analyze the results to obtain a prediction
  int act_index = PredictActivity(interpreter->output(0)->data.f);
 
  // Produce an output
  HandleOutput(error_reporter, act_index);//ca
  // Return time spent for inference
  Serial.println(infTime);
  infTime-=infTime;
  }
}
