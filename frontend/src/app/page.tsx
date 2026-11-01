"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import dynamic from "next/dynamic";
import { Header } from "../components/Header";
import { TriComparisonChart } from "../components/TriComparisonChart";
import { TerminalLog, ExecutionStep } from "../components/TerminalLog";
import { Play, RotateCw, Copy, Check, Bug, Sparkles, Layers, CornerDownLeft } from "lucide-react";

// Dynamically import Monaco Editor to avoid SSR hydration issues
const Editor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

const PRESETS = {
  fused_rms_norm: {
    title: "LLaMA-3 Fused RMSNorm",
    name: "fused_rms_norm",
    dimTag: "[32, 4096]",
    shapes: [[32, 4096]],
    dtype: "float16",
    code: `def rms_norm_eager(x, weight, eps=1e-6):
    # Eager RMSNorm allocates 3 intermediate HBM buffers:
    # 1. x.pow(2) elementwise square pass
    # 2. Mean reduction along hidden dimension (-1)
    # 3. Reciprocal square root (rsqrt) and scaling
    # KernelForge fuses these into a single SRAM tile pass.
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
    code: `def swiglu_eager(gate, up):
    # Eager SwiGLU executes two separate kernel launches:
    # 1. SiLU elementwise activation pass over gate tensor
    # 2. Vectorized Hadamard multiplication with up-projection
    # KernelForge fuses both operations into a single SRAM tile pass.
    return torch.nn.functional.silu(gate) * up
`
  },
  fused_softmax: {
    title: "Safe Online Softmax",
    name: "fused_softmax",
    dimTag: "[32, 4096]",
    shapes: [[32, 4096]],
    dtype: "float16",
    code: `def softmax_eager(scores):
    # Standard eager Softmax requires 3 passes over HBM:
    # 1. Row-max reduction for numerical stability (exp(x - max))
    # 2. Exponentiation and sum reduction for denominator
    # 3. Elementwise division (Softmax)
    # KernelForge synthesizes a 1-pass online Flash-Softmax.
    return torch.softmax(scores, dim=-1)
`
  }
};

export default function Home() {
  const [selectedPreset, setSelectedPreset] = useState<keyof typeof PRESETS>("fused_rms_norm");
  const [pytorchCode, setPytorchCode] = useState(PRESETS.fused_rms_norm.code);
  const [tritonCode, setTritonCode] = useState<string>("# Synthesized Triton (@triton.jit) kernel will appear here...\n");
  const [status, setStatus] = useState<"IDLE" | "COMPILING" | "BENCHMARKING" | "HEALING" | "COMPLETED" | "ERROR">("IDLE");
  const [steps, setSteps] = useState<ExecutionStep[]>([]);
  const [benchmark, setBenchmark] = useState<any>(null);
  const [tilingPlan, setTilingPlan] = useState<any>(null);
  const [triggerFaultDemo, setTriggerFaultDemo] = useState(false);
  const [copied, setCopied] = useState(false);

  const socketRef = useRef<WebSocket | null>(null);

  const handlePresetSelect = (key: keyof typeof PRESETS) => {
    setSelectedPreset(key);
    setPytorchCode(PRESETS[key].code);
    setTritonCode("# Click 'Synthesize & Compile' to generate custom Triton kernel...\n");
    setBenchmark(null);
    setTilingPlan(null);
    setSteps([]);
    setStatus("IDLE");
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
          if (resp.triton_source_code) {
            setTritonCode(resp.triton_source_code);
          }
          if (resp.benchmark) {
            setBenchmark(resp.benchmark);
          }
          if (resp.tiling_plan) {
            setTilingPlan(resp.tiling_plan);
          }
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
    } catch (e: any) {
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

  return (
    <div className="flex flex-col h-screen bg-[#090a0f] text-[#e2e8f0]">
      {/* Top Navigation */}
      <Header status={status} />

      {/* Main Workspace */}
      <div className="flex-1 flex flex-col p-4 gap-3.5 overflow-hidden">
        {/* Control Toolbar */}
        <div className="flex items-center justify-between bg-[#0e131d] border border-[#1f293d] px-4 py-2 rounded-lg font-mono text-xs">
          <div className="flex items-center gap-2">
            <span className="text-slate-400 font-bold mr-1">PRESET OPS:</span>
            {Object.entries(PRESETS).map(([key, val]) => (
              <button
                key={key}
                onClick={() => handlePresetSelect(key as keyof typeof PRESETS)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded transition-all text-xs font-medium ${
                  selectedPreset === key
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/60 font-bold shadow-sm shadow-cyan-950"
                    : "bg-slate-900/80 text-slate-300 hover:text-white hover:bg-slate-800 border border-slate-700 hover:border-slate-500"
                }`}
              >
                <span>{val.title}</span>
                <span className={`text-[10px] font-normal ${selectedPreset === key ? "text-cyan-400" : "text-slate-400"}`}>{val.dimTag}</span>
              </button>
            ))}
          </div>

          <div className="flex items-center gap-4">
            {/* Stress Test / Self-Healing Toggle (Engineering Reframe) */}
            <label className="flex items-center gap-2 cursor-pointer text-slate-300 hover:text-white select-none group">
              <input
                type="checkbox"
                checked={triggerFaultDemo}
                onChange={(e) => setTriggerFaultDemo(e.target.checked)}
                className="w-4 h-4 rounded bg-slate-900 border-slate-700 text-teal-500 focus:ring-0 focus:ring-offset-0 accent-teal-500 cursor-pointer"
              />
              <span className="flex items-center gap-1.5 text-slate-300 group-hover:text-teal-300 transition-colors font-medium">
                <Bug className="w-3.5 h-3.5 text-teal-400" />
                <span>Edge-Case Shape: <strong className="text-teal-300 font-semibold">N=4097</strong> (Unaligned Memory Stress)</span>
              </span>
            </label>

            {/* Synthesize Button with Hotkey Badge */}
            <button
              onClick={runSynthesis}
              disabled={status === "COMPILING" || status === "BENCHMARKING" || status === "HEALING"}
              className="flex items-center gap-2 px-4 py-2 rounded-md bg-gradient-to-r from-emerald-500 via-teal-400 to-cyan-500 hover:from-emerald-400 hover:to-cyan-400 text-black font-bold tracking-wider transition-all disabled:opacity-50 shadow-md shadow-emerald-500/20 active:scale-[0.98]"
            >
              {status === "COMPILING" || status === "BENCHMARKING" || status === "HEALING" ? (
                <>
                  <RotateCw className="w-4 h-4 animate-spin text-black" />
                  <span>PROCESSING...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-black" />
                  <span>SYNTHESIZE &amp; COMPILE</span>
                  <kbd className="hidden sm:inline-flex items-center gap-0.5 ml-1 px-1.5 py-0.5 rounded bg-black/25 text-[10px] font-mono text-black font-extrabold border border-black/10">
                    <CornerDownLeft className="w-2.5 h-2.5" /> Ctrl ↵
                  </kbd>
                </>
              )}
            </button>
          </div>
        </div>

        {/* 3-Column IDE Grid with Optimal 27% / 43% / 30% Width Split */}
        <div className="flex-1 flex gap-3.5 min-h-0">
          {/* Left Column (27%): Monaco PyTorch Input */}
          <div className="w-[27%] min-w-[270px] flex flex-col border border-[#1f293d] rounded-lg bg-[#0c1017] overflow-hidden">
            <div className="px-3 py-2 bg-[#0e1420] border-b border-[#1f293d] flex items-center justify-between text-xs font-mono text-slate-400">
              <span className="flex items-center gap-2">
                <Layers className="w-3.5 h-3.5 text-cyan-400" />
                PYTORCH EAGER INPUT
              </span>
              <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">REFERENCE OP</span>
            </div>
            <div className="flex-1 min-h-0">
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
                  automaticLayout: true
                }}
              />
            </div>
          </div>

          {/* Center Hero Column (43%): Monaco Synthesized Triton Output */}
          <div className="w-[43%] min-w-[420px] flex flex-col border border-[#1f293d] rounded-lg bg-[#0c1017] overflow-hidden shadow-lg shadow-black/40">
            <div className="px-3 py-2 bg-[#0e1420] border-b border-[#1f293d] flex items-center justify-between text-xs font-mono text-slate-400">
              <span className="flex items-center gap-2">
                <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
                <span className="font-semibold text-slate-200">SYNTHESIZED OPENAI TRITON</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-950/70 text-emerald-400 border border-emerald-800/50">@triton.jit</span>
              </span>
              <button
                onClick={handleCopyCode}
                className="flex items-center gap-1 px-2.5 py-1 rounded bg-[#182030] hover:bg-[#202c42] text-[10px] text-slate-300 transition-colors border border-slate-700/50 font-semibold"
              >
                {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                {copied ? "COPIED" : "COPY KERNEL"}
              </button>
            </div>
            <div className="flex-1 min-h-0">
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
                  wordWrap: "off" // Triton lines read clean with 43% width
                }}
              />
            </div>
          </div>

          {/* Right Column (30%): Tri-Comparison Latency & Telemetry */}
          <div className="w-[30%] min-w-[300px] flex flex-col min-h-0">
            <TriComparisonChart benchmark={benchmark} tilingPlan={tilingPlan} />
          </div>
        </div>

        {/* Bottom Full-Width Terminal: Multi-Agent WebSocket Log */}
        <div className="h-44 min-h-[160px]">
          <TerminalLog steps={steps} />
        </div>
      </div>
    </div>
  );
}
