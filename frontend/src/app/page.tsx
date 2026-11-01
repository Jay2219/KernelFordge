"use client";

import React, { useState, useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import { Header } from "../components/Header";
import { TriComparisonChart } from "../components/TriComparisonChart";
import { TerminalLog, ExecutionStep } from "../components/TerminalLog";
import { Play, RotateCw, Copy, Check, Bug, Sparkles, Layers } from "lucide-react";

// Dynamically import Monaco Editor to avoid SSR hydration issues
const Editor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

const PRESETS = {
  fused_rms_norm: {
    title: "LLaMA-3 Fused RMSNorm",
    name: "fused_rms_norm",
    shapes: [[32, 4096]],
    dtype: "float16",
    code: `def rms_norm_eager(x, weight, eps=1e-6):
    # Standard PyTorch allocates 3 intermediate HBM buffers:
    # 1. x.pow(2)
    # 2. variance reduction
    # 3. rsqrt inverse
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight
`
  },
  fused_swiglu: {
    title: "LLaMA-3 Fused SwiGLU",
    name: "fused_swiglu",
    shapes: [[32, 11008]],
    dtype: "float16",
    code: `def swiglu_eager(gate, up):
    # SwiGLU launches elementwise SiLU kernel followed by
    # intermediate multiplication with up-projection
    return torch.nn.functional.silu(gate) * up
`
  },
  fused_softmax: {
    title: "Safe Online Softmax",
    name: "fused_softmax",
    shapes: [[32, 4096]],
    dtype: "float16",
    code: `def softmax_eager(scores):
    # Softmax requires normalizer max tracking and sum reductions
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
  const [mockActive, setMockActive] = useState(true);

  const socketRef = useRef<WebSocket | null>(null);

  // Check backend status on mount
  useEffect(() => {
    fetch("http://localhost:8000/")
      .then((res) => res.json())
      .then((data) => {
        if (data && typeof data.mock_sandbox === "boolean") {
          setMockActive(data.mock_sandbox);
        }
      })
      .catch(() => setMockActive(true));
  }, []);

  const handlePresetSelect = (key: keyof typeof PRESETS) => {
    setSelectedPreset(key);
    setPytorchCode(PRESETS[key].code);
    setTritonCode("# Click 'Synthesize & Benchmark' to generate custom Triton kernel...\n");
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

  const runSynthesis = async () => {
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
      const ws = new WebSocket("ws://localhost:8000/ws/synthesize");
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
        // Fallback to REST API
        fallbackRestSynthesis(payload);
      };
    } catch {
      fallbackRestSynthesis(payload);
    }
  };

  const fallbackRestSynthesis = async (payload: any) => {
    try {
      const res = await fetch("http://localhost:8000/api/v1/synthesize", {
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

  return (
    <div className="flex flex-col h-screen bg-[#090a0f] text-[#e2e8f0]">
      {/* Top Navigation */}
      <Header status={status} mockMode={mockActive} />

      {/* Main Workspace */}
      <div className="flex-1 flex flex-col p-4 gap-4 overflow-hidden">
        {/* Control Toolbar */}
        <div className="flex items-center justify-between bg-[#0e131d] border border-[#1f293d] px-4 py-2.5 rounded-lg font-mono text-xs">
          <div className="flex items-center gap-2">
            <span className="text-slate-400 font-bold mr-1">PRESET OPS:</span>
            {Object.entries(PRESETS).map(([key, val]) => (
              <button
                key={key}
                onClick={() => handlePresetSelect(key as keyof typeof PRESETS)}
                className={`px-3 py-1.5 rounded transition-all ${
                  selectedPreset === key
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold"
                    : "bg-[#141b29] text-slate-400 hover:text-slate-200 border border-[#212c40]"
                }`}
              >
                {val.title}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 cursor-pointer text-slate-300 hover:text-white select-none">
              <input
                type="checkbox"
                checked={triggerFaultDemo}
                onChange={(e) => setTriggerFaultDemo(e.target.checked)}
                className="w-4 h-4 rounded border-slate-700 bg-slate-900 text-purple-500 focus:ring-purple-500"
              />
              <span className="flex items-center gap-1.5 text-purple-300 font-semibold">
                <Bug className="w-3.5 h-3.5 text-purple-400" />
                Trigger Fault Demo (Self-Healing)
              </span>
            </label>

            <button
              onClick={runSynthesis}
              disabled={status === "COMPILING" || status === "BENCHMARKING" || status === "HEALING"}
              className="flex items-center gap-2 px-5 py-2 rounded-md bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-400 hover:to-cyan-400 text-black font-bold tracking-wider transition-all disabled:opacity-50 shadow-lg shadow-emerald-500/20"
            >
              {status === "COMPILING" || status === "BENCHMARKING" || status === "HEALING" ? (
                <>
                  <RotateCw className="w-4 h-4 animate-spin text-black" />
                  PROCESSING...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-black" />
                  SYNTHESIZE &amp; COMPILE
                </>
              )}
            </button>
          </div>
        </div>

        {/* 3-Column IDE Grid */}
        <div className="flex-1 grid grid-cols-12 gap-4 min-h-0">
          {/* Left Column: Monaco PyTorch Input */}
          <div className="col-span-4 flex flex-col border border-[#1f293d] rounded-lg bg-[#0c1017] overflow-hidden">
            <div className="px-3 py-2 bg-[#0e1420] border-b border-[#1f293d] flex items-center justify-between text-xs font-mono text-slate-400">
              <span className="flex items-center gap-2">
                <Layers className="w-3.5 h-3.5 text-cyan-400" />
                PYTORCH EAGER SOURCE
              </span>
              <span className="text-[10px] text-slate-500">INPUT AST</span>
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

          {/* Center Column: Monaco Generated Triton Output */}
          <div className="col-span-5 flex flex-col border border-[#1f293d] rounded-lg bg-[#0c1017] overflow-hidden">
            <div className="px-3 py-2 bg-[#0e1420] border-b border-[#1f293d] flex items-center justify-between text-xs font-mono text-slate-400">
              <span className="flex items-center gap-2">
                <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
                SYNTHESIZED OPENAI TRITON (@triton.jit)
              </span>
              <button
                onClick={handleCopyCode}
                className="flex items-center gap-1 px-2 py-0.5 rounded bg-[#182030] hover:bg-[#202c42] text-[10px] text-slate-300 transition-colors"
              >
                {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                {copied ? "COPIED" : "COPY"}
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
                  automaticLayout: true
                }}
              />
            </div>
          </div>

          {/* Right Column: Tri-Comparison Latency & Telemetry */}
          <div className="col-span-3 flex flex-col min-h-0">
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
