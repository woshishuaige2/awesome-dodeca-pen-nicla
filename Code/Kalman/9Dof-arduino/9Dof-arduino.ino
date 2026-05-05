#include <Arduino_BHY2.h>
#include <ArduinoBLE.h>
#include <Nicla_System.h>

namespace {

// Stream the fused quaternion above the camera frame rate over USB serial.
constexpr unsigned long kSampleIntervalMs = 15;
constexpr float kSampleRateHz = 1000.0f / kSampleIntervalMs;
constexpr uint32_t kSensorLatencyMs = 1;
constexpr bool kPrintFusionDiagnostics = true;
constexpr bool kEnableBleNotifications = true;

enum class TransportMode {
  BleNotify,
  UsbSerial,
};

constexpr TransportMode kTransportMode = TransportMode::UsbSerial;
constexpr unsigned long kUsbBaudRate = 115200;

// BLE service/characteristic UUIDs stay stable, but the payload now carries a
// compact fused quaternion packet instead of raw IMU vectors.
constexpr char kDeviceName[] = "DPOINT";
constexpr char kServiceUuid[] = "19B10010-E8F2-537E-4F6C-D104768A1214";
constexpr char kImuCharacteristicUuid[] = "19B10013-E8F2-537E-4F6C-D104768A1214";

// Nicla-specific note:
// The original Xiao sketch powered the force sensor from a GPIO and sampled it
// with the nRF SAADC in differential mode. That path does not exist on Nicla.
// This rewrite keeps the pressure field in the BLE packet and reads it through
// the standard ADC path instead. If your sensor is externally powered, leave
// kPressureSensorPowerPin as -1. Otherwise, set it to the GPIO you use to feed
// the sensor and keep kPressureSensorPin wired to the analog output.
constexpr int kPressureSensorPin = A0;
constexpr int kPressureSensorPowerPin = -1;

struct QuaternionDataPacket {
  int16_t quat_wxyz[4];
  uint16_t pressure;
  uint16_t reserved;
  uint32_t timestamp_ms;
};

static_assert(sizeof(QuaternionDataPacket) == 16, "Unexpected QuaternionDataPacket size");

BLEService stylusService(kServiceUuid);
BLECharacteristic imuCharacteristic(
  kImuCharacteristicUuid,
  BLERead | BLENotify,
  sizeof(QuaternionDataPacket)
);

// The rotation vector uses the BHI260AP fusion output, which is the onboard
// quaternion estimate we want to stream instead of raw accel integration.
SensorQuaternion rotationVector(SENSOR_ID_RV);

unsigned long nextSampleAtMs = 0;
unsigned long diagnosticsWindowStartMs = 0;
unsigned long lastQuatChangeAtMs = 0;
uint32_t diagnosticsSentPacketCount = 0;
uint32_t diagnosticsQuatChangeCount = 0;
uint32_t diagnosticsRepeatedQuatCount = 0;
uint32_t diagnosticsQuatChangeIntervalSumMs = 0;
uint32_t diagnosticsUpdateCallCount = 0;
uint32_t diagnosticsSlowUpdateCount = 0;
uint32_t diagnosticsMaxUpdateMs = 0;
uint32_t diagnosticsUpdateDurationSumMs = 0;
uint32_t diagnosticsLoopCallCount = 0;
uint32_t diagnosticsMaxLoopMs = 0;
uint32_t diagnosticsSendCallCount = 0;
uint32_t diagnosticsMaxSendMs = 0;
uint32_t diagnosticsSendDurationSumMs = 0;
uint32_t diagnosticsMaxPressureMs = 0;
uint32_t diagnosticsMaxSerialMs = 0;
bool hasLastQuatPacket = false;
int16_t lastQuatPacket[4] = {0, 0, 0, 0};
uint16_t packetSequence = 0;

int16_t quantizeQuaternionComponent(float value) {
  const float clamped = constrain(value, -1.0f, 1.0f);
  return static_cast<int16_t>(clamped * 32767.0f);
}

bool quaternionPacketChanged(const QuaternionDataPacket& packet) {
  if (!hasLastQuatPacket) {
    return true;
  }
  for (size_t i = 0; i < 4; ++i) {
    if (packet.quat_wxyz[i] != lastQuatPacket[i]) {
      return true;
    }
  }
  return false;
}

void rememberQuaternionPacket(const QuaternionDataPacket& packet) {
  for (size_t i = 0; i < 4; ++i) {
    lastQuatPacket[i] = packet.quat_wxyz[i];
  }
  hasLastQuatPacket = true;
}

void printFusionDiagnosticsIfReady(unsigned long now) {
  if (!kPrintFusionDiagnostics) {
    return;
  }
  if (diagnosticsWindowStartMs == 0) {
    diagnosticsWindowStartMs = now;
    return;
  }
  if (now - diagnosticsWindowStartMs < 1000) {
    return;
  }

  const unsigned long windowMs = now - diagnosticsWindowStartMs;
  const float sentHz = diagnosticsSentPacketCount * 1000.0f / windowMs;
  const float quatChangeHz = diagnosticsQuatChangeCount * 1000.0f / windowMs;
  const float avgQuatChangeDtMs =
    diagnosticsQuatChangeCount > 1
      ? static_cast<float>(diagnosticsQuatChangeIntervalSumMs) / (diagnosticsQuatChangeCount - 1)
      : 0.0f;

  Serial.print("[Diag] sent_hz=");
  Serial.print(sentHz, 2);
  Serial.print(" quat_change_hz=");
  Serial.print(quatChangeHz, 2);
  Serial.print(" repeated_packets=");
  Serial.print(diagnosticsRepeatedQuatCount);
  Serial.print(" avg_quat_change_dt_ms=");
  Serial.print(avgQuatChangeDtMs, 2);
  Serial.print(" update_calls=");
  Serial.print(diagnosticsUpdateCallCount);
  Serial.print(" slow_updates=");
  Serial.print(diagnosticsSlowUpdateCount);
  Serial.print(" avg_update_ms=");
  Serial.print(
    diagnosticsUpdateCallCount > 0
      ? static_cast<float>(diagnosticsUpdateDurationSumMs) / diagnosticsUpdateCallCount
      : 0.0f,
    2
  );
  Serial.print(" max_update_ms=");
  Serial.print(diagnosticsMaxUpdateMs);
  Serial.print(" loop_calls=");
  Serial.print(diagnosticsLoopCallCount);
  Serial.print(" max_loop_ms=");
  Serial.print(diagnosticsMaxLoopMs);
  Serial.print(" send_calls=");
  Serial.print(diagnosticsSendCallCount);
  Serial.print(" avg_send_ms=");
  Serial.print(
    diagnosticsSendCallCount > 0
      ? static_cast<float>(diagnosticsSendDurationSumMs) / diagnosticsSendCallCount
      : 0.0f,
    2
  );
  Serial.print(" max_send_ms=");
  Serial.print(diagnosticsMaxSendMs);
  Serial.print(" max_pressure_ms=");
  Serial.print(diagnosticsMaxPressureMs);
  Serial.print(" max_serial_ms=");
  Serial.println(diagnosticsMaxSerialMs);

  diagnosticsWindowStartMs = now;
  diagnosticsSentPacketCount = 0;
  diagnosticsQuatChangeCount = 0;
  diagnosticsRepeatedQuatCount = 0;
  diagnosticsQuatChangeIntervalSumMs = 0;
  diagnosticsUpdateCallCount = 0;
  diagnosticsSlowUpdateCount = 0;
  diagnosticsMaxUpdateMs = 0;
  diagnosticsUpdateDurationSumMs = 0;
  diagnosticsLoopCallCount = 0;
  diagnosticsMaxLoopMs = 0;
  diagnosticsSendCallCount = 0;
  diagnosticsMaxSendMs = 0;
  diagnosticsSendDurationSumMs = 0;
  diagnosticsMaxPressureMs = 0;
  diagnosticsMaxSerialMs = 0;
}

void fatalBlink() {
  while (true) {
    nicla::leds.setColor(red);
    delay(150);
    nicla::leds.setColor(off);
    delay(150);
  }
}

void configurePressureSensor() {
  analogReadResolution(12);

  if (kPressureSensorPowerPin >= 0) {
    pinMode(kPressureSensorPowerPin, OUTPUT);
    digitalWrite(kPressureSensorPowerPin, HIGH);
    delay(5);
  }
}

uint16_t readPressure() {
  const unsigned long startMs = millis();
  const int raw = analogRead(kPressureSensorPin);
  const long scaled = map(raw, 0, 4095, 0, 65535);
  const unsigned long durationMs = millis() - startMs;
  if (durationMs > diagnosticsMaxPressureMs) {
    diagnosticsMaxPressureMs = durationMs;
  }
  return static_cast<uint16_t>(constrain(scaled, 0L, 65535L));
}

void configureImu() {
  // We manage BLE ourselves in this sketch, so keep the BHY2 side in
  // standalone mode and only enable the fused rotation vector output.
  BHY2.begin(NICLA_STANDALONE);
  const bool rotationStarted = rotationVector.begin(kSampleRateHz, kSensorLatencyMs);
  Serial.print("rotation_vector_begin=");
  Serial.println(rotationStarted ? "true" : "false");
  SensorConfig config = rotationVector.getConfiguration();
  Serial.print("rotation_config sample_rate=");
  Serial.print(config.sample_rate, 2);
  Serial.print(" latency_ms=");
  Serial.print(config.latency);
  Serial.print(" range=");
  Serial.println(config.range);
}

void configureBle() {
  if (!BLE.begin()) {
    Serial.println("BLE.begin() failed");
    fatalBlink();
  }

  BLE.setLocalName(kDeviceName);
  BLE.setDeviceName(kDeviceName);
  BLE.setAdvertisedService(stylusService);

  stylusService.addCharacteristic(imuCharacteristic);
  BLE.addService(stylusService);

  const QuaternionDataPacket initialPacket = {};
  imuCharacteristic.writeValue(
    reinterpret_cast<const uint8_t*>(&initialPacket),
    sizeof(initialPacket)
  );

  BLE.advertise();
  if (!kEnableBleNotifications) {
    Serial.println("BLE notifications disabled for fusion-rate debug mode");
  }
}

void writeUsbPacket(const QuaternionDataPacket& packet) {
  const unsigned long startMs = millis();
  Serial.print("Q,");
  Serial.print(packet.reserved);
  Serial.print(",");
  Serial.print(packet.timestamp_ms);
  for (size_t i = 0; i < 4; ++i) {
    Serial.print(",");
    Serial.print(packet.quat_wxyz[i]);
  }
  Serial.print(",");
  Serial.println(packet.pressure);
  const unsigned long durationMs = millis() - startMs;
  if (durationMs > diagnosticsMaxSerialMs) {
    diagnosticsMaxSerialMs = durationMs;
  }
}

void sendPacketIfReady() {
  const unsigned long now = millis();
  if (static_cast<long>(now - nextSampleAtMs) < 0) {
    return;
  }
  const unsigned long sendStartMs = millis();
  nextSampleAtMs = now + kSampleIntervalMs;

  QuaternionDataPacket packet = {};
  packet.quat_wxyz[0] = quantizeQuaternionComponent(rotationVector.w());
  packet.quat_wxyz[1] = quantizeQuaternionComponent(rotationVector.x());
  packet.quat_wxyz[2] = quantizeQuaternionComponent(rotationVector.y());
  packet.quat_wxyz[3] = quantizeQuaternionComponent(rotationVector.z());
  packet.pressure = readPressure();
  packet.reserved = packetSequence++;
  packet.timestamp_ms = now;

  diagnosticsSentPacketCount++;
  if (quaternionPacketChanged(packet)) {
    diagnosticsQuatChangeCount++;
    if (lastQuatChangeAtMs != 0) {
      diagnosticsQuatChangeIntervalSumMs += now - lastQuatChangeAtMs;
    }
    lastQuatChangeAtMs = now;
    rememberQuaternionPacket(packet);
  } else {
    diagnosticsRepeatedQuatCount++;
  }

  if (kTransportMode == TransportMode::UsbSerial) {
    writeUsbPacket(packet);
  } else if (kEnableBleNotifications) {
    imuCharacteristic.writeValue(
      reinterpret_cast<const uint8_t*>(&packet),
      sizeof(packet)
    );
  }
  const unsigned long sendDurationMs = millis() - sendStartMs;
  diagnosticsSendCallCount++;
  diagnosticsSendDurationSumMs += sendDurationMs;
  if (sendDurationMs > diagnosticsMaxSendMs) {
    diagnosticsMaxSendMs = sendDurationMs;
  }
  printFusionDiagnosticsIfReady(now);
}

}  // namespace

void setup() {
  Serial.begin(kUsbBaudRate);
  delay(500);

  nicla::begin();
  nicla::leds.begin();
  nicla::leds.setColor(blue);

  configurePressureSensor();
  configureImu();
  if (kTransportMode == TransportMode::BleNotify) {
    configureBle();
  } else {
    nicla::leds.setColor(green);
  }

  nextSampleAtMs = millis();
  diagnosticsWindowStartMs = nextSampleAtMs;
  Serial.println("Nicla stylus streamer ready");
  Serial.print("sample_interval_ms=");
  Serial.print(kSampleIntervalMs);
  Serial.print(" sample_rate_hz=");
  Serial.println(kSampleRateHz, 2);
}

void loop() {
  const unsigned long loopStartMs = millis();
  if (kTransportMode == TransportMode::BleNotify) {
    BLE.poll();
  }
  const unsigned long updateStartMs = millis();
  BHY2.update();
  const unsigned long updateDurationMs = millis() - updateStartMs;
  diagnosticsUpdateCallCount++;
  diagnosticsUpdateDurationSumMs += updateDurationMs;
  if (updateDurationMs > diagnosticsMaxUpdateMs) {
    diagnosticsMaxUpdateMs = updateDurationMs;
  }
  if (updateDurationMs > 20) {
    diagnosticsSlowUpdateCount++;
  }

  if (kTransportMode == TransportMode::UsbSerial) {
    sendPacketIfReady();
  } else if (BLE.connected()) {
    nicla::leds.setColor(green);
    sendPacketIfReady();
  } else {
    nicla::leds.setColor(blue);
  }
  const unsigned long loopDurationMs = millis() - loopStartMs;
  diagnosticsLoopCallCount++;
  if (loopDurationMs > diagnosticsMaxLoopMs) {
    diagnosticsMaxLoopMs = loopDurationMs;
  }
}
