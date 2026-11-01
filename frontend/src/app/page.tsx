"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import dynamic from "next/dynamic";
import { TriComparisonChart } from "../components/TriComparisonChart";
import { TerminalLog, ExecutionStep } from "../components/TerminalLog";
import { VERIFIED_GOLDEN_RUNS, GoldenRun } from "../data/goldenRuns";
import { 
  Zap, Cpu, Check, Copy, Play, RotateCw, 
  Layers, Sparkles, CornerDownLeft, Bug, CheckCircle 
} from "lucide-react";

// Dynamically import Monaco Editor to avoid SSR hydration issues
const Editor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

interface HealthState {
  status: string;
  gpu_pool: {
    status: string;
    target: string;
    latency_ms: number;
    peak_bandwidth_gbs: number;
    provider: string;
  };
  models: {
    meta_api: {
      status: string;
      model: string;
      latency_ms: number;
      provider: string;
    };
    baseten: {
      status: string;
      model: string;
      latency_ms: number;
      provider: string;
    };
  };
}

const PRESETS = {
  fused_rms_norm: {
    title: "LLaMA-3 Fused RMSNorm",
    name: "fused_rms_norm",
    dimTag: "[32, 4096]",
    shapes: [[32, 4096]],
    dtype: "float16",
    code: `def rms_norm_eager(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # PyTorch Eager allocates 3 intermediate HBM buffers:
    # 1. Elementwise square: x.pow(2)
    # 2. Reduction across dim: variance = mean(-1)
    # 3. Scale & norm: x * rsqrt(variance + eps) * weight
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight
`
  },
  fused_swiglu: {
    title: "LLaMA-3 Fused SwiGLU",
    name: "fused_swiglu",
    dimTag: "[32, 11008]",
    shapes: [[32, 11008]],
    dtype: "float16",
    code: `def swiglu_eager(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    # Eager SwiGLU requires 2 separate kernel launches:
    # 1. SiLU: silu(gate) = gate * sigmoid(gate)
    # 2. Multiplication with up-projection
    return torch.nn.functional.silu(gate) * up
`
  },
  fused_softmax: {
    title: "Safe Online Softmax",
    name: "fused_softmax",
    dimTag: "[32, 4096]",
    shapes: [[32, 4096]],
    dtype: "float16",
    code: `def softmax_eager(scores: torch.Tensor) -> torch.Tensor:
    # Eager Softmax triggers 3 roundtrips to HBM memory:
    # 1. Row max tracking for numerical stability
    # 2. Exponentiation & sum-reduction for denominator
    # 3. Elementwise scaling by inverse normalizer
    return torch.softmax(scores, dim=-1)
`
  }
};

interface VerifiedRunInfo {
  isVerified: boolean;
  runId?: string;
  device?: string;
  timestamp?: string;
}

export default function KernelForgeWorkbench() {
  const [selectedPreset, setSelectedPreset] = useState<keyof typeof PRESETS>("fused_rms_norm");
  const initialGolden = VERIFIED_GOLDEN_RUNS.fused_rms_norm;
  
  const [pytorchCode, setPytorchCode] = useState(initialGolden.pytorch_source);
  const [tritonCode, setTritonCode] = useState<string>(initialGolden.triton_source_code);
  const [status, setStatus] = useState<"IDLE" | "COMPILING" | "BENCHMARKING" | "HEALING" | "COMPLETED" | "ERROR">("COMPLETED");
  const [steps, setSteps] = useState<ExecutionStep[]>(initialGolden.steps as any);
  const [benchmark, setBenchmark] = useState<any>(initialGolden.benchmark);
  const [tilingPlan, setTilingPlan] = useState<any>(initialGolden.tiling_plan);
  const [triggerFaultDemo, setTriggerFaultDemo] = useState(false);
  const [copied, setCopied] = useState(false);
  const [health, setHealth] = useState<HealthState | null>(null);
  const [verifiedRunInfo, setVerifiedRunInfo] = useState<VerifiedRunInfo | null>({
    isVerified: true,
    runId: initialGolden.run_id,
    device: initialGolden.hardware.gpu_name,
    timestamp: initialGolden.timestamp
  });

  const socketRef = useRef<WebSocket | null>(null);

  // Poll pre-flight infrastructure diagnostic health endpoint every 30s
  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/health`);
        if (res.ok) {
          const data = await res.json();
          setHealth(data);
        }
      } catch {
        // Retain fallback telemetry
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handlePresetSelect = (key: keyof typeof PRESETS) => {
    setSelectedPreset(key);
    const golden = VERIFIED_GOLDEN_RUNS[key];
    if (golden) {
      setPytorchCode(golden.pytorch_source);
      setTritonCode(golden.triton_source_code);
      setBenchmark(golden.benchmark);
      setTilingPlan(golden.tiling_plan);
      setSteps(golden.steps as any);
      setStatus("COMPLETED");
      setVerifiedRunInfo({
        isVerified: true,
        runId: golden.run_id,
        device: golden.hardware.gpu_name,
        timestamp: golden.timestamp
      });
    } else {
      setPytorchCode(PRESETS[key].code);
      setTritonCode("# Click 'Synthesize & Compile' to generate custom Triton kernel...\n");
      setBenchmark(null);
      setTilingPlan(null);
      setSteps([]);
      setStatus("IDLE");
      setVerifiedRunInfo(null);
    }
  };

  const handleCopyCode = () => {
    navigator.clipboard.writeText(tritonCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const runSynthesis = useCallback(async () => {
    if (status === "COMPILING" || status === "BENCHMARKING" || status === "HEALING") return;

    setStatus("COMPILING");
    setSteps([]);
    setBenchmark(null);
    setTilingPlan(null);
    setVerifiedRunInfo({ isVerified: false });

    const payload = {
      session_id: `session-${Date.now()}`,
      operation_name: selectedPreset,
      pytorch_source: pytorchCode,
      input_tensor_shapes: PRESETS[selectedPreset].shapes,
      dtype: PRESETS[selectedPreset].dtype,
      trigger_fault_demo: triggerFaultDemo
    };

    // Attempt WebSocket streaming
    try {
      const ws = new WebSocket(`${WS_BASE}/ws/synthesize`);
      socketRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify(payload));
      };

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === "STEP_UPDATE" && msg.step) {
          const step: ExecutionStep = msg.step;
          setSteps((prev) => [...prev, step]);

          if (step.step_name === "TILING_PLAN" && step.data) {
            setTilingPlan(step.data);
          }
          if (step.code_snippet) {
            setTritonCode(step.code_snippet);
          }
          if (step.step_name === "SANDBOX_DISPATCH") {
            setStatus(step.status === "FAILED" ? "HEALING" : "BENCHMARKING");
          }
          if (step.step_name === "REFLECTION_REPAIR") {
            setStatus("HEALING");
          }
        }

        if (msg.type === "SYNTHESIS_COMPLETE" && msg.response) {
          const resp = msg.response;
          if (resp.triton_source_code) setTritonCode(resp.triton_source_code);
          if (resp.benchmark) setBenchmark(resp.benchmark);
          if (resp.tiling_plan) setTilingPlan(resp.tiling_plan);
          setStatus(resp.status === "COMPLETED" ? "COMPLETED" : "ERROR");
          ws.close();
        }

        if (msg.type === "ERROR") {
          setStatus("ERROR");
          ws.close();
        }
      };

      ws.onerror = () => {
        fallbackRestSynthesis(payload);
      };
    } catch {
      fallbackRestSynthesis(payload);
    }
  }, [status, selectedPreset, pytorchCode, triggerFaultDemo]);

  const fallbackRestSynthesis = async (payload: any) => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/synthesize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.triton_source_code) setTritonCode(data.triton_source_code);
      if (data.benchmark) setBenchmark(data.benchmark);
      if (data.tiling_plan) setTilingPlan(data.tiling_plan);
      if (data.steps) setSteps(data.steps);
      setStatus(data.status === "COMPLETED" ? "COMPLETED" : "ERROR");
    } catch {
      setStatus("ERROR");
    }
  };

  // Keyboard shortcut: Ctrl + Enter to synthesize
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        runSynthesis();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [runSynthesis]);

  const isBusy = status === "COMPILING" || status === "BENCHMARKING" || status === "HEALING";
  const modalLatency = health?.gpu_pool?.latency_ms ?? 24;
  const basetenLatency = health?.models?.baseten?.latency_ms ?? 42;
  const metaLatency = health?.models?.meta_api?.latency_ms ?? 31;

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-[#07090E] text-slate-200 font-sans select-none">
      
      {/* 1. SLIM COMMAND HEADER (Fixed 52px) */}
      <header className="h-[52px] border-b border-white/[0.08] bg-[#0A0D14] px-4 flex items-center justify-between shrink-0 z-10">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
              <Zap className="w-4 h-4 fill-cyan-400/20" />
            </div>
            <span className="font-semibold text-white tracking-tight text-sm">KernelForge</span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white/[0.06] text-slate-400 border border-white/[0.04]">
              v0.1.0
            </span>
          </div>

          <div className="h-4 w-px bg-white/[0.08] mx-1" />

          {/* Infrastructure Diagnostic Badges */}
          <div className="flex items-center gap-2 text-xs">
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-white/[0.03] border border-white/[0.06] text-slate-300">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span>A10G <span className="text-slate-500 text-[11px]">(600 GB/s)</span></span>
            </div>

            <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-cyan-500/10 border border-cyan-500/25 text-cyan-300 text-[11px]">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
              <span className="font-medium">Muse Wonder + Code</span>
              <span className="text-[10px] font-mono text-cyan-400/80 font-medium">({metaLatency}ms)</span>
            </div>

            <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-500/10 border border-amber-500/30 text-amber-300 font-medium text-[11px]">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
              <span>Baseten Open-Weights</span>
              <span className="text-[10px] font-mono text-amber-400/80 font-medium">({basetenLatency}ms)</span>
            </div>

            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-[11px]">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              <span>Modal A10G Live</span>
              <span className="text-[10px] font-mono text-emerald-400/90 font-medium">({modalLatency}ms)</span>
            </div>
          </div>
        </div>

        {/* Action CTA */}
        <div className="flex items-center gap-3">
          <button
            onClick={runSynthesis}
            disabled={isBusy}
            className="flex items-center gap-2 px-4 py-1.5 rounded-lg bg-gradient-to-r from-cyan-500 via-teal-400 to-emerald-400 hover:from-cyan-400 hover:to-emerald-300 text-black font-semibold text-xs transition-all shadow-lg shadow-cyan-500/20 active:scale-95 disabled:opacity-50"
          >
            {isBusy ? (
              <>
                <RotateCw className="w-3.5 h-3.5 animate-spin text-black" />
                <span>Compiling on A10G...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-black" />
                <span>Synthesize &amp; Compile</span>
                <span className="text-[10px] bg-black/20 px-1 rounded font-mono ml-1 font-bold">Ctrl ↵</span>
              </>
            )}
          </button>
        </div>
      </header>

      {/* 2. COMPACT PRESET CHIP BAR (Fixed 38px) */}
      <div className="h-[38px] border-b border-white/[0.06] bg-[#07090E] px-4 flex items-center justify-between text-xs shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-slate-400 text-[11px] font-medium mr-1">Presets:</span>
          {Object.entries(PRESETS).map(([key, val]) => (
            <button
              key={key}
              onClick={() => handlePresetSelect(key as keyof typeof PRESETS)}
              className={`px-2.5 py-1 rounded-md transition-all text-xs font-medium flex items-center ${
                selectedPreset === key
                  ? "bg-cyan-500/15 border border-cyan-500/40 text-cyan-300 font-medium"
                  : "text-slate-400 hover:text-slate-200 hover:bg-white/[0.04]"
              }`}
            >
              <span>{val.title}</span>
              <span className={`ml-2 px-1.5 py-0.5 rounded text-[10px] font-mono ${
                selectedPreset === key
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                  : "bg-white/[0.06] text-slate-400 border border-white/[0.04]"
              }`}>
                {val.dimTag}
              </span>
            </button>
          ))}
        </div>

        {/* Custom Toggle Switch for Edge-Case */}
        <label className="flex items-center gap-2 cursor-pointer text-slate-300 hover:text-white transition-colors">
          <input
            type="checkbox"
            checked={triggerFaultDemo}
            onChange={(e) => setTriggerFaultDemo(e.target.checked)}
            className="w-3.5 h-3.5 rounded bg-slate-900 border-slate-700 text-teal-500 focus:ring-0 accent-cyan-500 cursor-pointer"
          />
          <span className="text-[11px] font-medium flex items-center gap-1">
            <Bug className="w-3 h-3 text-purple-400" />
            <span>Stress Test: <strong className="font-mono text-purple-400 font-semibold">N=4097</strong> (Unaligned Boundary Mask)</span>
          </span>
        </label>
      </div>

      {/* 3. MAIN WORKSPACE (Flex Row: 28% / 42% / 30% Exact Proportions) */}
      <main className="flex-1 flex overflow-hidden p-2 gap-2">
        
        {/* COLUMN 1: PyTorch Eager Input (28%) */}
        <section className="w-[28%] flex flex-col rounded-lg border border-white/[0.08] bg-[#0A0D14] overflow-hidden">
          <div className="h-8 border-b border-white/[0.06] bg-white/[0.02] px-3 flex items-center justify-between text-xs text-slate-400 shrink-0 font-mono">
            <span className="flex items-center gap-2 text-slate-300 font-medium">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              PyTorch Eager Source
            </span>
            <span className="text-[10px] text-slate-500 font-semibold tracking-wider">REFERENCE OP</span>
          </div>
          <div className="flex-1 min-h-0 bg-[#080B10]">
            <Editor
              height="100%"
              defaultLanguage="python"
              value={pytorchCode}
              onChange={(val) => setPytorchCode(val || "")}
              theme="vs-dark"
              options={{
                minimap: { enabled: false },
                fontSize: 12,
                fontFamily: "JetBrains Mono, monospace",
                lineNumbers: "on",
                scrollBeyondLastLine: false,
                automaticLayout: true,
                wordWrap: "on",
                wrappingIndent: "indent",
                lineNumbersMinChars: 3
              }}
            />
          </div>
        </section>

        {/* COLUMN 2: Synthesized Triton Hero (42%) */}
        <section className="w-[42%] flex flex-col rounded-lg border border-white/[0.08] bg-[#0A0D14] overflow-hidden shadow-lg shadow-black/40">
          <div className="h-8 border-b border-white/[0.06] bg-white/[0.02] px-3 flex items-center justify-between text-xs shrink-0 font-mono">
            <div className="flex items-center gap-2">
              <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
              <span className="font-semibold text-slate-200">Synthesized OpenAI Triton</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">
                @triton.jit
              </span>
              {verifiedRunInfo?.isVerified && (
                <span className="hidden sm:flex items-center gap-1 text-[10px] px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 font-mono font-medium">
                  <CheckCircle className="w-3 h-3 text-emerald-400" />
                  <span>VERIFIED: {verifiedRunInfo.runId}</span>
                </span>
              )}
            </div>
            <button
              onClick={handleCopyCode}
              className="flex items-center gap-1 text-[11px] text-slate-400 hover:text-white transition-colors"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? "Copied" : "Copy Kernel"}</span>
            </button>
          </div>
          <div className="flex-1 min-h-0 bg-[#080B10]">
            <Editor
              height="100%"
              defaultLanguage="python"
              value={tritonCode}
              theme="vs-dark"
              options={{
                readOnly: true,
                minimap: { enabled: false },
                fontSize: 12,
                fontFamily: "JetBrains Mono, monospace",
                lineNumbers: "on",
                scrollBeyondLastLine: false,
                automaticLayout: true,
                wordWrap: "off" // Spacious 42% width ensures equations don't wrap
              }}
            />
          </div>
        </section>

        {/* COLUMN 3: Telemetry Inspector with 2 Tabs (30%) */}
        <section className="w-[30%] flex flex-col rounded-lg overflow-hidden">
          <TriComparisonChart 
            benchmark={benchmark} 
            tilingPlan={tilingPlan} 
            verifiedRunInfo={verifiedRunInfo} 
          />
        </section>
      </main>

      {/* 4. AGENT COMPILER TRACE DRAWER (Fixed 160px) */}
      <footer className="h-[160px] border-t border-white/[0.08] bg-[#06080D] flex flex-col shrink-0">
        <TerminalLog steps={steps} />
      </footer>

    </div>
  );
}
