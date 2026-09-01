"use client";

import React, { useState, useEffect, useRef } from "react";
import { 
  Smartphone, 
  Activity, 
  MapPin, 
  ShieldAlert, 
  CheckCircle, 
  AlertTriangle, 
  Play, 
  Square, 
  UploadCloud, 
  Loader2, 
  X,
  Compass
} from "lucide-react";
import { api } from "@/lib/api";

interface RealTimeSensorCaptureProps {
  caseId: string;
  onSuccess: (summary: any) => void;
  onCancel: () => void;
}

export function RealTimeSensorCapture({ caseId, onSuccess, onCancel }: RealTimeSensorCaptureProps) {
  // Support & Permissions
  const [motionSupported] = useState<boolean>(() => typeof window !== "undefined" && "DeviceMotionEvent" in window);
  const [orientationSupported] = useState<boolean>(() => typeof window !== "undefined" && "DeviceOrientationEvent" in window);
  const [motionPermission, setMotionPermission] = useState<"prompt" | "granted" | "denied">(() => {
    if (typeof window !== "undefined" && "DeviceMotionEvent" in window) {
      if (typeof (DeviceMotionEvent as any).requestPermission === "function") return "prompt";
      return "granted";
    }
    return "prompt";
  });
  const [locationChoice, setLocationChoice] = useState<"none" | "requested" | "enabled" | "skipped">("none");
  const [locationCoords, setLocationCoords] = useState<{ lat: number; lon: number; speed: number | null } | null>(null);

  // Live Hardware Ticks Detection
  const [receivingMotionEvents, setReceivingMotionEvents] = useState<boolean>(false);

  // Recording State
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recordingComplete, setRecordingComplete] = useState<boolean>(false);
  const [recordingTime, setRecordingTime] = useState<number>(0);

  // Live Readings
  const [currentAccel, setCurrentAccel] = useState<{ x: number; y: number; z: number; mag: number } | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [currentGyro, setCurrentGyro] = useState<{ alpha: number; beta: number; gamma: number }>({ alpha: 0, beta: 0, gamma: 0 });

  // Samples Array
  const samplesRef = useRef<any[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = useRef<number | null>(null);

  const [uploading, setUploading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Request iOS Motion Permission
  const requestMotionPermission = async () => {
    try {
      if (typeof (DeviceMotionEvent as any).requestPermission === "function") {
        const res = await (DeviceMotionEvent as any).requestPermission();
        if (res === "granted") {
          setMotionPermission("granted");
          if (typeof (DeviceOrientationEvent as any).requestPermission === "function") {
            await (DeviceOrientationEvent as any).requestPermission();
          }
        } else {
          setMotionPermission("denied");
        }
      } else {
        setMotionPermission("granted");
      }
    } catch (err) {
      console.warn("Motion permission request failed:", err);
      setMotionPermission("denied");
    }
  };

  // Location Enable Flow
  const handleEnableLocation = () => {
    setLocationChoice("requested");
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          setLocationCoords({
            lat: pos.coords.latitude,
            lon: pos.coords.longitude,
            speed: pos.coords.speed
          });
          setLocationChoice("enabled");
        },
        (err) => {
          console.warn("Location permission denied or unavailable:", err.message);
          setLocationChoice("skipped");
        },
        { enableHighAccuracy: true, timeout: 5000 }
      );
    } else {
      setLocationChoice("skipped");
    }
  };

  const handleSkipLocation = () => {
    setLocationChoice("skipped");
  };

  // Sensor Event Listeners
  useEffect(() => {
    if (motionPermission !== "granted") return;

    let tickCount = 0;

    const handleMotion = (e: DeviceMotionEvent) => {
      tickCount++;
      if (tickCount > 2 && !receivingMotionEvents) {
        setReceivingMotionEvents(true);
      }

      const acc = e.accelerationIncludingGravity || e.acceleration;
      const x = acc?.x;
      const y = acc?.y;
      const z = acc?.z;
      if (x == null || y == null || z == null) {
        return;
      }
      const mag = Math.sqrt(x * x + y * y + z * z);

      setCurrentAccel({
        x: Number(x.toFixed(2)),
        y: Number(y.toFixed(2)),
        z: Number(z.toFixed(2)),
        mag: Number(mag.toFixed(2))
      });

      if (isRecording) {
        const ts = Date.now();
        samplesRef.current.push({
          timestamp: ts,
          acceleration_x: e.acceleration?.x ?? null,
          acceleration_y: e.acceleration?.y ?? null,
          acceleration_z: e.acceleration?.z ?? null,
          acceleration_gravity_x: acc?.x ?? null,
          acceleration_gravity_y: acc?.y ?? null,
          acceleration_gravity_z: acc?.z ?? null,
          rotation_rate_alpha: e.rotationRate?.alpha ?? null,
          rotation_rate_beta: e.rotationRate?.beta ?? null,
          rotation_rate_gamma: e.rotationRate?.gamma ?? null,
          latitude: locationCoords?.lat ?? null,
          longitude: locationCoords?.lon ?? null,
          speed: locationCoords?.speed ?? null
        });
      }
    };

    const handleOrientation = (e: DeviceOrientationEvent) => {
      const alpha = e.alpha ?? 0;
      const beta = e.beta ?? 0;
      const gamma = e.gamma ?? 0;
      setCurrentGyro({
        alpha: Number(alpha.toFixed(1)),
        beta: Number(beta.toFixed(1)),
        gamma: Number(gamma.toFixed(1))
      });
    };

    window.addEventListener("devicemotion", handleMotion);
    window.addEventListener("deviceorientation", handleOrientation);

    return () => {
      window.removeEventListener("devicemotion", handleMotion);
      window.removeEventListener("deviceorientation", handleOrientation);
    };
  }, [motionPermission, isRecording, receivingMotionEvents, locationCoords]);

  const [capturedCount, setCapturedCount] = useState<number>(0);
  const [finalRate, setFinalRate] = useState<number>(0);

  // Recording Timer
  const startRecording = () => {
    samplesRef.current = [];
    setIsRecording(true);
    setRecordingComplete(false);
    setRecordingTime(0);
    setCapturedCount(0);
    setFinalRate(0);
    // eslint-disable-next-line react-hooks/purity
    startTimeRef.current = Date.now();

    timerRef.current = setInterval(() => {
      if (startTimeRef.current) {
        const elapsed = (Date.now() - startTimeRef.current) / 1000;
        setRecordingTime(Number(elapsed.toFixed(1)));

        // Max limit safety stop (60 seconds)
        if (elapsed >= 60) {
          stopRecording();
        }
      }
    }, 100);
  };

  const stopRecording = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    const count = samplesRef.current.length;
    const duration = recordingTime > 0 ? recordingTime : 0.1;
    const rate = count > 1 ? Number(((count - 1) / duration).toFixed(1)) : 0;
    
    setCapturedCount(count);
    setFinalRate(rate);
    setIsRecording(false);
    setRecordingComplete(true);
  };

  // Submit Live Sensor Payload to Backend
  const handleAnalyzeLiveSensor = async () => {
    if (samplesRef.current.length < 10) {
      setError("Recording too short. Please capture at least 10 sensor samples (minimum 2-3 seconds of motion).");
      return;
    }

    setUploading(true);
    setError(null);

    const recordedDuration = recordingTime > 0 ? recordingTime : 0.1;

    const payload = {
      source_type: "live",
      device_metadata: {
        user_agent: typeof navigator !== "undefined" ? navigator.userAgent : "unknown",
        platform: typeof navigator !== "undefined" ? navigator.platform : "unknown",
        motion_supported: motionSupported,
        gyroscope_supported: orientationSupported,
        location_enabled: locationChoice === "enabled"
      },
      recording_duration_seconds: recordedDuration,
      observed_sampling_rate_hz: finalRate,
      samples: samplesRef.current
    };

    try {
      const res = await api.uploadLiveSensor(caseId, payload);
      onSuccess(res.summary);
    } catch (err: any) {
      setError(err.message || "Failed to process live sensor recording.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="dash-card p-6 space-y-6 max-w-2xl mx-auto border border-blue-800/40 bg-[#0B1224] rounded-2xl shadow-2xl relative text-xs">
      {/* Header & Cancel */}
      <div className="flex justify-between items-start border-b border-[#26324A] pb-3">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-blue-600/20 border border-blue-500/40 text-blue-400 rounded-xl glow-blue">
            <Smartphone className="h-6 w-6" />
          </div>
          <div>
            <h3 className="text-base font-extrabold text-white flex items-center space-x-2">
              <span>Real-Time Device Sensor Capture</span>
              <span className="px-2 py-0.5 bg-emerald-950/80 text-emerald-400 border border-emerald-800 text-[10px] font-mono rounded">
                LIVE API
              </span>
            </h3>
            <p className="text-[11px] text-slate-400">
              Captures physical accelerometer, gyroscope, and optional location telemetry directly from your browser.
            </p>
          </div>
        </div>
        <button onClick={onCancel} className="p-1 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800">
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Global Error Notice */}
      {error && (
        <div className="p-3 bg-red-950/40 border border-red-800 rounded-xl flex items-center space-x-2 text-red-300 text-xs">
          <AlertTriangle className="h-4 w-4 text-red-400 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Safety Notice */}
      <div className="p-3 bg-amber-950/20 border border-amber-800/40 rounded-xl flex items-start space-x-2.5 text-[11px] text-amber-300">
        <ShieldAlert className="h-4 w-4 text-amber-400 flex-shrink-0 mt-0.5" />
        <div>
          <strong className="block font-bold text-amber-400">Research Safety &amp; Testing Disclaimer</strong>
          <span>
            This feature measures motion hardware data for technical evaluation. It is for research demonstration only and does not provide emergency diagnosis. Do not attempt intentional falls or dangerous collisions. Test normal movements like walking or tilting the device.
          </span>
        </div>
      </div>

      {/* Desktop / Unsupported Device Fallback */}
      {!motionSupported ? (
        <div className="p-5 bg-slate-900/80 border border-slate-800 rounded-xl text-center space-y-3">
          <AlertTriangle className="h-8 w-8 text-amber-400 mx-auto animate-bounce" />
          <h4 className="text-sm font-bold text-white">Real-Time Sensors Unavailable on This Device / Browser</h4>
          <p className="text-[11px] text-slate-400 max-w-md mx-auto">
            Your current browser or desktop environment does not expose mobile motion hardware APIs (accelerometer/gyroscope).
          </p>
          <div className="pt-2">
            <button
              onClick={onCancel}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs rounded-xl"
            >
              Return to Upload, Demo, or Simulation Modes
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-5">
          {/* Permission Flow */}
          {motionPermission === "prompt" && (
            <div className="p-4 bg-blue-950/30 border border-blue-800/60 rounded-xl flex items-center justify-between">
              <div className="space-y-0.5">
                <span className="font-bold text-white block">Device Motion Permission Required</span>
                <p className="text-[11px] text-slate-400">iOS Safari requires an explicit button tap to access motion hardware.</p>
              </div>
              <button
                onClick={requestMotionPermission}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl shadow"
              >
                Grant Motion Access
              </button>
            </div>
          )}

          {/* Location Permission Flow (Correction 4: Explicit Optional Location) */}
          <div className="p-3.5 bg-slate-900/60 border border-slate-800 rounded-xl space-y-2">
            <div className="flex justify-between items-center">
              <div className="flex items-center space-x-2">
                <MapPin className="h-4 w-4 text-emerald-400" />
                <span className="font-bold text-slate-200">Optional Location Telemetry (GPS)</span>
              </div>
              <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded ${
                locationChoice === "enabled" ? "bg-emerald-950 text-emerald-400 border border-emerald-800" : "bg-slate-800 text-slate-400"
              }`}>
                {locationChoice === "enabled" ? "GPS ENABLED" : (locationChoice === "skipped" ? "GPS SKIPPED" : "GPS OPTIONAL")}
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              Location data is strictly optional. Your coordinates will only be recorded if explicitly enabled. Motion recording works fully without GPS.
            </p>
            {locationChoice === "none" && (
              <div className="flex space-x-2 pt-1">
                <button
                  type="button"
                  onClick={handleEnableLocation}
                  className="px-3 py-1.5 bg-emerald-600/80 hover:bg-emerald-500 text-white font-bold text-[11px] rounded-lg flex items-center space-x-1"
                >
                  <MapPin className="h-3 w-3" />
                  <span>ENABLE LOCATION</span>
                </button>
                <button
                  type="button"
                  onClick={handleSkipLocation}
                  className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold text-[11px] rounded-lg"
                >
                  CONTINUE WITHOUT LOCATION
                </button>
              </div>
            )}
          </div>

          {/* Hardware Connection Status (Correction 9) */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 text-[11px]">
            <div className="p-2.5 bg-slate-950/60 border border-slate-800 rounded-xl flex items-center space-x-2">
              <Activity className={`h-4 w-4 ${receivingMotionEvents ? "text-emerald-400" : "text-amber-400 animate-pulse"}`} />
              <div>
                <span className="text-slate-400 block text-[10px]">Accelerometer</span>
                <strong className={receivingMotionEvents ? "text-emerald-400" : "text-amber-400"}>
                  {receivingMotionEvents ? "Receiving Ticks" : "Waiting for Motion"}
                </strong>
              </div>
            </div>
            <div className="p-2.5 bg-slate-950/60 border border-slate-800 rounded-xl flex items-center space-x-2">
              <Compass className="h-4 w-4 text-purple-400" />
              <div>
                <span className="text-slate-400 block text-[10px]">Gyroscope</span>
                <strong className={orientationSupported ? "text-purple-400" : "text-slate-500"}>
                  {orientationSupported ? "Connected" : "Unavailable"}
                </strong>
              </div>
            </div>
            <div className="p-2.5 bg-slate-950/60 border border-slate-800 rounded-xl flex items-center space-x-2">
              <MapPin className="h-4 w-4 text-blue-400" />
              <div>
                <span className="text-slate-400 block text-[10px]">GPS Telemetry</span>
                <strong className={locationChoice === "enabled" ? "text-emerald-400" : "text-slate-500"}>
                  {locationChoice === "enabled" ? "Active" : "Skipped"}
                </strong>
              </div>
            </div>
          </div>

          {/* Live Telemetry Display */}
          <div className="p-4 bg-[#080D1C] border border-[#26324A] rounded-xl space-y-3">
            <div className="flex justify-between items-center border-b border-slate-800 pb-2">
              <span className="font-bold text-slate-300">Live Hardware Readings</span>
              {isRecording && (
                <span className="flex items-center space-x-1.5 text-red-400 font-bold animate-pulse text-[11px]">
                  <span className="h-2 w-2 rounded-full bg-red-500"></span>
                  <span>RECORDING LIVE ({recordingTime}s)</span>
                </span>
              )}
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
              <div className="p-2 bg-slate-900/80 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-500 block">X (Accel)</span>
                <span className="font-mono font-bold text-emerald-400">{currentAccel == null ? "FEATURE_MISSING" : `${currentAccel.x} m/s²`}</span>
              </div>
              <div className="p-2 bg-slate-900/80 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-500 block">Y (Accel)</span>
                <span className="font-mono font-bold text-emerald-400">{currentAccel == null ? "FEATURE_MISSING" : `${currentAccel.y} m/s²`}</span>
              </div>
              <div className="p-2 bg-slate-900/80 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-500 block">Z (Accel)</span>
                <span className="font-mono font-bold text-emerald-400">{currentAccel == null ? "FEATURE_MISSING" : `${currentAccel.z} m/s²`}</span>
              </div>
              <div className="p-2 bg-blue-950/40 rounded-lg border border-blue-800/60">
                <span className="text-[10px] text-blue-400 block font-bold">Magnitude</span>
                <span className="font-mono font-bold text-blue-300">{currentAccel == null ? "FEATURE_MISSING" : `${currentAccel.mag} m/s²`}</span>
              </div>
            </div>
          </div>

          {/* Recording Controls */}
          <div className="flex items-center justify-between pt-2">
            {!isRecording && !recordingComplete && (
              <button
                type="button"
                onClick={startRecording}
                disabled={motionPermission !== "granted"}
                className="w-full py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-extrabold rounded-xl flex items-center justify-center space-x-2 shadow-lg disabled:opacity-50 transition-all glow-blue"
              >
                <Play className="h-4 w-4 fill-current" />
                <span>START REAL-TIME RECORDING</span>
              </button>
            )}

            {isRecording && (
              <button
                type="button"
                onClick={stopRecording}
                className="w-full py-3 bg-red-600 hover:bg-red-500 text-white font-extrabold rounded-xl flex items-center justify-center space-x-2 shadow-lg transition-all animate-pulse"
              >
                <Square className="h-4 w-4 fill-current" />
                <span>STOP RECORDING ({recordingTime}s)</span>
              </button>
            )}

            {recordingComplete && !isRecording && (
              <div className="w-full space-y-3">
                <div className="p-3 bg-emerald-950/30 border border-emerald-800/60 rounded-xl flex items-center justify-between text-emerald-300">
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-emerald-400" />
                    <span><strong>Recording Complete:</strong> {recordingTime} seconds ({capturedCount} samples captured, ~{finalRate} Hz)</span>
                  </div>
                  <button
                    onClick={startRecording}
                    className="text-[11px] underline hover:text-emerald-200"
                  >
                    Re-record
                  </button>
                </div>

                <button
                  type="button"
                  onClick={handleAnalyzeLiveSensor}
                  disabled={uploading}
                  className="w-full py-3.5 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white font-extrabold rounded-xl flex items-center justify-center space-x-2 shadow-xl disabled:opacity-50 transition-all"
                >
                  {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <UploadCloud className="h-4 w-4" />}
                  <span>ANALYZE SENSOR DATA (PASS TO PIPELINE)</span>
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
