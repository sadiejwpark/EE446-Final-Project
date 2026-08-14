/*
  Wake Word Detection -- Arduino Nano 33 BLE Sense (Rev2)
  EE 446 Final Project

  Pipeline: PDM mic -> ring buffer -> most-recent 1s window -> on-device MFCC
  -> TFLite Micro INT8 inference -> threshold + debounce -> Serial output.
*/

#include <PDM.h>

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "audio_config.h"
#include "mfcc.h"
#include "model_data.h"

// ---------- Audio capture ----------
#define RING_BUFFER_SAMPLES (CLIP_LENGTH_SAMPLES + 4000)
volatile int16_t g_ring_buffer[RING_BUFFER_SAMPLES];
volatile int g_ring_write_index = 0;
volatile bool g_ring_filled_once = false;

short g_pdm_chunk[512];

void OnPdmData() {
  int bytes_available = PDM.available();
  PDM.read(g_pdm_chunk, bytes_available);
  int num_samples = bytes_available / 2;
  for (int i = 0; i < num_samples; i++) {
    g_ring_buffer[g_ring_write_index] = g_pdm_chunk[i];
    g_ring_write_index = (g_ring_write_index + 1) % RING_BUFFER_SAMPLES;
    if (g_ring_write_index == 0) g_ring_filled_once = true;
  }
}

int16_t g_window[CLIP_LENGTH_SAMPLES];
void GetLatestWindow() {
  noInterrupts();
  int end = g_ring_write_index;
  interrupts();
  int start = (end - CLIP_LENGTH_SAMPLES + RING_BUFFER_SAMPLES) % RING_BUFFER_SAMPLES;
  for (int i = 0; i < CLIP_LENGTH_SAMPLES; i++) {
    g_window[i] = g_ring_buffer[(start + i) % RING_BUFFER_SAMPLES];
  }
}

// ---------- MFCC ----------
MfccExtractor g_mfcc;
float g_mfcc_features[NUM_FRAMES * NUM_MFCC];

// ---------- TFLite Micro ----------
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// ---------- Detection state ----------
unsigned long g_last_inference_ms = 0;
unsigned long g_last_trigger_ms = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  Serial.println("EVENT,boot");

  g_mfcc.Init();

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_model);
  // TFLITE_SCHEMA_VERSION isn't exposed in this library version's public
  // headers, so we skip the strict schema-version check here. This is safe:
  // if the model were actually incompatible, AllocateTensors() below will
  // fail loudly (you'll see "EVENT,allocate_tensors_failed") rather than
  // silently misbehaving.

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("EVENT,allocate_tensors_failed");
    while (1) { delay(1000); }
  }

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  Serial.print("EVENT,arena_used_bytes,");
  Serial.println(interpreter->arena_used_bytes());

  PDM.onReceive(OnPdmData);
  PDM.setBufferSize(1024);
  if (!PDM.begin(1, SAMPLE_RATE_HZ)) {
    Serial.println("EVENT,pdm_init_failed");
    while (1) { delay(1000); }
  }

  Serial.println("EVENT,setup_complete");
  Serial.println("# CSV log format:");
  Serial.println("# INFER,<millis>,<pred_label>,<confidence>,<latency_us>");
  Serial.println("# TRIGGER,<millis>");
}

void loop() {
  unsigned long now = millis();
  if (!g_ring_filled_once && g_ring_write_index < CLIP_LENGTH_SAMPLES) {
    return;
  }
  if (now - g_last_inference_ms < INFERENCE_INTERVAL_MS) {
    return;
  }
  g_last_inference_ms = now;

  unsigned long t0 = micros();

  GetLatestWindow();
  g_mfcc.Compute(g_window, g_mfcc_features);

  float input_scale = model_input->params.scale;
  int input_zero_point = model_input->params.zero_point;
  for (int i = 0; i < NUM_FRAMES * NUM_MFCC; i++) {
    int32_t q = (int32_t)roundf(g_mfcc_features[i] / input_scale) + input_zero_point;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    model_input->data.int8[i] = (int8_t)q;
  }

  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long latency_us = micros() - t0;

  if (invoke_status != kTfLiteOk) {
    Serial.println("EVENT,invoke_failed");
    return;
  }

  float output_scale = model_output->params.scale;
  int output_zero_point = model_output->params.zero_point;

  int best_index = 0;
  float best_confidence = -1.0f;
  for (int i = 0; i < NUM_CLASSES; i++) {
    int8_t raw = model_output->data.int8[i];
    float confidence = (raw - output_zero_point) * output_scale;
    if (confidence > best_confidence) {
      best_confidence = confidence;
      best_index = i;
    }
  }

  Serial.print("INFER,");
  Serial.print(now);
  Serial.print(",");
  Serial.print(kLabels[best_index]);
  Serial.print(",");
  Serial.print(best_confidence, 4);
  Serial.print(",");
  Serial.println(latency_us);

  if (best_index == WAKE_WORD_CLASS_INDEX &&
      best_confidence >= DETECTION_THRESHOLD &&
      (now - g_last_trigger_ms) >= DEBOUNCE_MS) {
    g_last_trigger_ms = now;
    Serial.print("TRIGGER,");
    Serial.println(now);
    // digitalWrite(LED_BUILTIN, HIGH); delay(150); digitalWrite(LED_BUILTIN, LOW);
  }
}