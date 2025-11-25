// Include main library header file
#include <MicroTFLite.h>

// Include network weights and test data
#include "net.h"      // network weights (seizure_model[])
#include "testdata.h" // test data (with x_test_cut[40][1024])

#define N_INPUTS 1024 // length of array for each sample
#define N_OUTPUTS 2   // binary classification -> seizure/no seizure

// The Tensor Arena memory area is used by TensorFlow Lite to store input, 
// output and intermediate tensors. You may need to adjust this size based 
// on your model's requirements.
constexpr int kTensorArenaSize = 35*1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

void setup() {
    Serial.begin(9600);
    while (!Serial);
    
    delay(5000);
    
    Serial.println("Seizure Detection Inference Example");
    Serial.println("Initializing TensorFlow Lite Micro Interpreter...");
    
    // Give error if model has a problem
    if (!ModelInit(seizure_model, tensor_arena, kTensorArenaSize)) {
        Serial.println("Model initialization failed!");
        while (true) delay(5000);
    }
    
    Serial.println("Model initialization done.");
    
    // Print model metadata & tensor information.
    ModelPrintMetadata(); delay(500);
    ModelPrintInputTensorDimensions(); delay(500);
    ModelPrintOutputTensorDimensions(); delay(500);
    
    Serial.println("");
    Serial.println("Starting seizure detection on test data...");
}

void loop() {
    float x_test[N_INPUTS] = {0};
    int j = 0;

    Serial.println("Id \t Category \t Time"); // table header

    for (j = 0; j < 40; j++) {
        // Copy test data to input array
        for (int i = 0; i < N_INPUTS; i++) {
            x_test[i] = x_test_cut[j][i]; // use data from testdata.h
        }

        uint32_t start = micros(); //start inference timer

        // Set input data for each element in the tensor
        for (int i = 0; i < N_INPUTS; i++) {
            if (!ModelSetInput(x_test[i], i)) {
                Serial.print("Failed to set input at index ");
                Serial.println(i);
                while (true) delay(1000);
            }
        }

        // Run inference
        if (!ModelRunInference()) {
            Serial.println("RunInference Failed!");
            while (true) delay(1000);
        }

        uint32_t timeit = micros() - start; // stop inference timer

        // Get possibility of each output
        float output0 = ModelGetOutput(0);
        float output1 = ModelGetOutput(1);
        
        // Determine predicted class (0 or 1)
        int predicted_class = (output1 > output0) ? 1 : 0;

        Serial.print(j + 1);
        Serial.print("\t");
        Serial.print(predicted_class);
        Serial.print("\t");
        Serial.print(timeit / 1000000.0, 6); // Convert to seconds
        Serial.println();

        delay(1000);
    }
    
    Serial.println("Test completed. Restarting in 5 seconds...");
    delay(5000);
}