#include <Arduino_BHY2.h>
#include <ArduinoBLE.h>
#include <Nicla_System.h>

namespace {

constexpr unsigned long kSampleIntervalMs = 7;
constexpr float kSampleRateHz = 1000.0f / kSampleIntervalMs;
constexpr uint32_t kSensorLatencyMs = 1;
constexpr bool kPrintFusionDiagnostics = false;
constexpr bool kEnableBleNotifications = true;

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
bool hasLastQuatPacket = false;
int16_t lastQuatPacket[4] = {0, 0, 0, 0};

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
  Serial.println(avgQuatChangeDtMs, 2);

  diagnosticsWindowStartMs = now;
  diagnosticsSentPacketCount = 0;
  diagnosticsQuatChangeCount = 0;
  diagnosticsRepeatedQuatCount = 0;
  diagnosticsQuatChangeIntervalSumMs = 0;
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
  const int raw = analogRead(kPressureSensorPin);
  const long scaled = map(raw, 0, 4095, 0, 65535);
  return static_cast<uint16_t>(constrain(scaled, 0L, 65535L));
}

void configureImu() {
  // We manage BLE ourselves in this sketch, so keep the BHY2 side in
  // standalone mode and only enable the fused rotation vector output.
  BHY2.begin(NICLA_STANDALONE);
  rotationVector.begin(kSampleRateHz, kSensorLatencyMs);
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

void sendPacketIfReady() {
  const unsigned long now = millis();
  if (static_cast<long>(now - nextSampleAtMs) < 0) {
    return;
  }
  nextSampleAtMs = now + kSampleIntervalMs;

  BHY2.update();

  QuaternionDataPacket packet = {};
  packet.quat_wxyz[0] = quantizeQuaternionComponent(rotationVector.w());
  packet.quat_wxyz[1] = quantizeQuaternionComponent(rotationVector.x());
  packet.quat_wxyz[2] = quantizeQuaternionComponent(rotationVector.y());
  packet.quat_wxyz[3] = quantizeQuaternionComponent(rotationVector.z());
  packet.pressure = readPressure();
  packet.reserved = 0;
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

  if (kEnableBleNotifications) {
    imuCharacteristic.writeValue(
      reinterpret_cast<const uint8_t*>(&packet),
      sizeof(packet)
    );
  }
  printFusionDiagnosticsIfReady(now);
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(500);

  nicla::begin();
  nicla::leds.begin();
  nicla::leds.setColor(blue);

  configurePressureSensor();
  configureImu();
  configureBle();

  nextSampleAtMs = millis();
  diagnosticsWindowStartMs = nextSampleAtMs;
  Serial.println("Nicla stylus streamer ready");
}

void loop() {
  BLE.poll();
  BHY2.update();

  if (BLE.connected()) {
    nicla::leds.setColor(green);
    sendPacketIfReady();
  } else {
    nicla::leds.setColor(blue);
  }
}
