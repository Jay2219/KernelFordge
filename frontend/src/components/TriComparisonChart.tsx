"use client";

import React from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Zap, Activity, CheckCircle, ShieldAlert, Cpu } from "lucide-react";

interface BenchmarkData {
  pytorch_eager_us: number;
  pytorch_compile_us: number;
  triton_kernel_us: number;
  speedup_vs_eager: number;
  speedup_vs_compile: number;
  achieved_bandwidth_gbs: number;
  bandwidth_saturation_pct: number;
  numerical_parity_verified: boolean;
  max_absolute_error?: number;
  max_relative_error?: number;
}

interface TriComparisonChartProps {
  benchmark: BenchmarkData | null;
  tilingPlan?: any;
}

// Authentic 2D Log-Log Roofline Model for NVIDIA A10G
// Memory Bandwidth Ceiling: 600 GB/s (0.6 TB/s)
// Peak Compute Ceiling: 31.2 TFLOPS (FP32/TF32 Tensor Cores)
// Operational Knee: 31.2 / 0.6 = 52.0 FLOPs/Byte
const A10GRooflineSVG: React.FC<{
  achievedBandwidth?: number;
  saturationPct?: number;
  isStandby?: boolean;
}> = ({ achievedBandwidth = 0, saturationPct = 0, isStandby = false }) => {
  // SVG Viewport: 340 x 95
  // Plot Area: x from 38 to 325, y from 12 to 78
  // Knee point: x = 233, y = 24
  const kneeX = 233;
  const kneeY = 24;
  const startX = 54;
  const startY = 78;
  const endX = 325;
  const endY = 24;

  // Operating point for fused memory-bound kernel (Softmax/RMSNorm/SwiGLU @ Arithmetic Intensity ~1.0 FLOP/Byte)
  const opX = 110;
  // Dynamic Y for Triton operating point based on saturation (ceiling y=65, baseline eager y=75)
  const tritonY = isStandby ? 66 : Math.max(26, 75 - (saturationPct / 100) * 11);
  const eagerY = 75;

  return (
    <div className="w-full bg-[#0b0e15] rounded-md border border-[#1e2738] p-2.5">
      <div className="flex items-center justify-between text-[10px] text-slate-400 mb-1">
        <span className="font-semibold text-cyan-400 flex items-center gap-1">
          <Cpu className="w-3 h-3 text-cyan-400" /> A10G 2D ROOFLINE MODEL
        </span>
        <span className="text-slate-500 font-mono">
          BW: <strong className="text-slate-300">600 GB/s</strong> | Knee: <strong className="text-slate-300">52 FL/B</strong>
        </span>
      </div>

      <svg viewBox="0 0 340 95" className="w-full h-[95px] select-none font-mono">
        <defs>
          <linearGradient id="rooflineGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.25" />
            <stop offset="60%" stopColor="#06b6d4" stopOpacity="0.08" />
            <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.0" />
          </linearGradient>
          <filter id="glowGreen" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2.5" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Shaded Area Under Roofline */}
        <polygon
          points={`${startX},${startY} ${kneeX},${kneeY} ${endX},${endY} ${endX},${startY}`}
          fill="url(#rooflineGradient)"
        />

        {/* Gridlines */}
        <line x1="38" y1="78" x2="325" y2="78" stroke="#1e293b" strokeWidth="1" />
        <line x1="38" y1="48" x2="325" y2="48" stroke="#161f30" strokeWidth="0.8" strokeDasharray="3 3" />
        <line x1="38" y1="24" x2="325" y2="24" stroke="#161f30" strokeWidth="0.8" strokeDasharray="3 3" />
        <line x1="38" y1="12" x2="38" y2="78" stroke="#1e293b" strokeWidth="1" />
        <line x1="110" y1="12" x2="110" y2="78" stroke="#161f30" strokeWidth="0.8" strokeDasharray="3 3" />
        <line x1="233" y1="12" x2="233" y2="78" stroke="#161f30" strokeWidth="0.8" strokeDasharray="3 3" />

        {/* The Roofline Ceilings */}
        {/* 1. Sloping Memory Ceiling (600 GB/s) */}
        <line
          x1={startX}
          y1={startY}
          x2={kneeX}
          y2={kneeY}
          stroke="#06b6d4"
          strokeWidth="2"
          strokeLinecap="round"
        />
        {/* 2. Flat Compute Ceiling (31.2 TFLOPS) */}
        <line
          x1={kneeX}
          y1={kneeY}
          x2={endX}
          y2={endY}
          stroke="#38bdf8"
          strokeWidth="2"
          strokeLinecap="round"
        />

        {/* Knee / Ridge Point Indicator */}
        <circle cx={kneeX} cy={kneeY} r="3" fill="#38bdf8" />
        <text x={kneeX - 4} y={kneeY - 6} fill="#7dd3fc" fontSize="7.5" fontWeight="bold">
          Knee (52 FL/B)
        </text>

        {/* Ceilings Labels */}
        <text x="62" y="44" fill="#06b6d4" fontSize="7.5" fontWeight="bold" transform="rotate(-18, 62, 44)">
          Memory Ceiling: 600 GB/s
        </text>
        <text x="240" y="20" fill="#38bdf8" fontSize="7" fontWeight="bold">
          Compute: 31.2 TF
        </text>

        {/* Axis Ticks & Labels */}
        <text x="35" y="80" fill="#64748b" fontSize="7" textAnchor="end">0.1</text>
        <text x="35" y="50" fill="#64748b" fontSize="7" textAnchor="end">1.0</text>
        <text x="35" y="26" fill="#64748b" fontSize="7" textAnchor="end">31</text>
        <text x="32" y="16" fill="#64748b" fontSize="6.5" textAnchor="end">TF</text>

        <text x="54" y="88" fill="#64748b" fontSize="7" textAnchor="middle">0.1</text>
        <text x="110" y="88" fill="#38bdf8" fontSize="7.5" fontWeight="bold" textAnchor="middle">1.0</text>
        <text x="180" y="88" fill="#64748b" fontSize="7" textAnchor="middle">10</text>
        <text x="233" y="88" fill="#7dd3fc" fontSize="7" textAnchor="middle">52</text>
        <text x="325" y="88" fill="#64748b" fontSize="7" textAnchor="middle">10³</text>
        <text x="280" y="93" fill="#64748b" fontSize="6.5">FLOPs/Byte →</text>

        {/* Operating Points: Eager vs KernelForge Triton */}
        {/* Eager PyTorch Operating Point (Red) */}
        <circle cx={opX} cy={eagerY} r="3" fill="#f43f5e" />
        <text x={opX + 5} y={eagerY + 2.5} fill="#f43f5e" fontSize="7">
          Eager
        </text>

        {/* Upward Acceleration Vector */}
        <line
          x1={opX}
          y1={eagerY - 3}
          x2={opX}
          y2={tritonY + 4}
          stroke="#10b981"
          strokeWidth="1.2"
          strokeDasharray="2 2"
        />
        <polygon
          points={`${opX},${tritonY + 2} ${opX - 2.5},${tritonY + 6} ${opX + 2.5},${tritonY + 6}`}
          fill="#10b981"
        />

        {/* KernelForge Triton Operating Point (Glowing Green) */}
        <circle
          cx={opX}
          cy={tritonY}
          r="4.5"
          fill="#10b981"
          filter="url(#glowGreen)"
        />
        <circle cx={opX} cy={tritonY} r="2" fill="#ffffff" />
        <text
          x={opX + 7}
          y={tritonY + 2.5}
          fill="#34d399"
          fontSize="7.5"
          fontWeight="bold"
        >
          {isStandby ? "Triton Target (~88% HBM)" : `Triton (${saturationPct}% HBM)`}
        </text>

        {/* Region Domain Tag */}
        <text x="60" y="70" fill="#475569" fontSize="6.5" fontStyle="italic">
          ◀ MEMORY-BOUND ZONE
        </text>
      </svg>
    </div>
  );
};

export const TriComparisonChart: React.FC<TriComparisonChartProps> = ({ benchmark, tilingPlan }) => {
  if (!benchmark) {
    return (
      <div className="h-full flex flex-col justify-between p-4 border border-[#1f293d] rounded-lg bg-[#0e131d]/90 font-mono text-xs">
        {/* Top Banner Skeleton */}
        <div className="flex items-center justify-between pb-2.5 border-b border-[#1f293d]/60">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-900/80 border border-slate-800 text-slate-400 text-[11px] font-semibold">
            <Activity className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
            <span>HARDWARE-COALESCED ROOFLINE</span>
          </div>
          <div className="text-right text-slate-400 text-[11px] font-medium">
            <span>0.00x vs Eager</span>
            <span className="mx-2 text-slate-700">|</span>
            <span>0.00x vs Compile</span>
          </div>
        </div>

        {/* 2D Roofline Model (Authentic Standby Visualization) */}
        <div className="my-1">
          <A10GRooflineSVG isStandby={true} saturationPct={88} />
        </div>

        {/* Ghosted Latency Bars */}
        <div className="flex flex-col gap-2 my-1">
          <div className="text-[11px] text-slate-400 flex items-center justify-between font-semibold">
            <span>KERNEL EXECUTION LATENCY</span>
            <span className="text-slate-500 font-normal">do_bench() µs</span>
          </div>

          {/* Bar 1: PyTorch Eager Skeleton */}
          <div className="space-y-0.5">
            <div className="flex justify-between text-[10px] text-slate-400">
              <span className="text-rose-400/80 font-semibold">PyTorch Eager</span>
              <span>--.- µs</span>
            </div>
            <div className="w-full h-4 bg-[#121722] rounded border border-rose-950/40 overflow-hidden relative">
              <div className="w-[85%] h-full bg-rose-500/10 border-r border-rose-500/30" />
            </div>
          </div>

          {/* Bar 2: torch.compile Skeleton */}
          <div className="space-y-0.5">
            <div className="flex justify-between text-[10px] text-slate-400">
              <span className="text-amber-400/80 font-semibold">torch.compile (Inductor)</span>
              <span>--.- µs</span>
            </div>
            <div className="w-full h-4 bg-[#121722] rounded border border-amber-950/40 overflow-hidden relative">
              <div className="w-[45%] h-full bg-amber-500/10 border-r border-amber-500/30" />
            </div>
          </div>

          {/* Bar 3: KernelForge Triton Skeleton */}
          <div className="space-y-0.5">
            <div className="flex justify-between text-[10px] text-slate-400">
              <span className="text-emerald-400/90 font-semibold">KernelForge Triton</span>
              <span className="text-emerald-400/80">TARGET: &lt;120 µs</span>
            </div>
            <div className="w-full h-4 bg-[#121722] rounded border border-emerald-950/40 overflow-hidden relative">
              <div className="w-[30%] h-full bg-emerald-500/15 border-r border-emerald-400/40" />
            </div>
          </div>
        </div>

        {/* Ghosted Bandwidth Saturation Meter */}
        <div className="pt-2 border-t border-[#1f293d]/60">
          <div className="flex items-center justify-between mb-1 text-[11px]">
            <span className="text-slate-400 flex items-center gap-1.5 font-semibold">
              <Zap className="w-3.5 h-3.5 text-cyan-400" /> MEMORY BUS SATURATION
            </span>
            <span className="text-slate-400">--.- GB/s (--%)</span>
          </div>
          <div className="w-full h-2 bg-[#121722] rounded-full overflow-hidden border border-slate-800">
            <div className="w-[88%] h-full bg-cyan-500/10 border-r border-cyan-500/20" />
          </div>
          <div className="flex justify-between text-[10px] text-slate-500 mt-1">
            <span>0 GB/s</span>
            <span>Target A10G Peak: 600 GB/s</span>
          </div>
        </div>

        {/* Ghosted Hardware Tiling Grid */}
        <div className="pt-2 border-t border-[#1f293d]/60 grid grid-cols-3 gap-2 text-[11px] text-center text-slate-400">
          <div className="bg-[#121724]/80 p-1.5 rounded border border-slate-800/80">
            <span className="text-slate-500 block text-[9px]">BLOCK SIZE</span>
            <span className="text-slate-300 font-bold">1024</span>
          </div>
          <div className="bg-[#121724]/80 p-1.5 rounded border border-slate-800/80">
            <span className="text-slate-500 block text-[9px]">WARP COUNT</span>
            <span className="text-slate-300 font-bold">4 Warps</span>
          </div>
          <div className="bg-[#121724]/80 p-1.5 rounded border border-slate-800/80">
            <span className="text-slate-500 block text-[9px]">STAGES</span>
            <span className="text-slate-300 font-bold">3 Stages</span>
          </div>
        </div>
      </div>
    );
  }

  const chartData = [
    {
      name: "PyTorch Eager",
      latency: benchmark.pytorch_eager_us,
      color: "#f43f5e",
      label: `${benchmark.pytorch_eager_us.toFixed(1)} µs`
    },
    {
      name: "torch.compile",
      latency: benchmark.pytorch_compile_us,
      color: "#eab308",
      label: `${benchmark.pytorch_compile_us.toFixed(1)} µs`
    },
    {
      name: "KernelForge Triton",
      latency: benchmark.triton_kernel_us,
      color: "#10b981",
      label: `${benchmark.triton_kernel_us.toFixed(1)} µs`
    }
  ];

  return (
    <div className="h-full flex flex-col justify-between p-4 border border-[#1f293d] rounded-lg bg-[#0e131d]/90 font-mono text-xs">
      {/* Top Banner: Verification Badge & Speedup Highlights */}
      <div className="flex items-center justify-between pb-2.5 border-b border-[#1f293d]">
        <div className="flex items-center gap-2">
          {benchmark.numerical_parity_verified ? (
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-xs font-bold shadow-sm shadow-emerald-500/10">
              <CheckCircle className="w-3.5 h-3.5" /> PARITY: PASS (rtol=1e-3)
            </span>
          ) : (
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-rose-500/10 text-rose-400 border border-rose-500/30 text-xs font-bold">
              <ShieldAlert className="w-3.5 h-3.5" /> NUMERICAL MISMATCH
            </span>
          )}
        </div>
        <div className="text-right">
          <span className="text-emerald-400 font-bold text-sm">+{benchmark.speedup_vs_eager.toFixed(2)}x</span>
          <span className="text-slate-400 text-[10px] ml-1">vs Eager</span>
          <span className="text-cyan-300 font-semibold text-xs ml-2.5">+{benchmark.speedup_vs_compile.toFixed(2)}x</span>
          <span className="text-slate-400 text-[10px] ml-1">vs Compile</span>
        </div>
      </div>

      {/* 2D Roofline Model (Active Performance State) */}
      <div className="my-1">
        <A10GRooflineSVG
          achievedBandwidth={benchmark.achieved_bandwidth_gbs}
          saturationPct={benchmark.bandwidth_saturation_pct}
          isStandby={false}
        />
      </div>

      {/* Latency Bar Chart */}
      <div className="flex flex-col justify-center my-1">
        <div className="text-[11px] text-slate-400 mb-1 flex items-center justify-between font-semibold">
          <span>KERNEL LATENCY COMPARISON</span>
          <span className="text-slate-500 font-normal">do_bench() µs</span>
        </div>
        <div className="h-[120px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ top: 2, right: 35, left: 65, bottom: 2 }}>
              <XAxis type="number" stroke="#475569" tick={{ fontSize: 10 }} unit="µs" />
              <YAxis type="category" dataKey="name" stroke="#cbd5e1" tick={{ fontSize: 10 }} width={80} />
              <Tooltip
                contentStyle={{ backgroundColor: "#0b0f17", borderColor: "#1e293b", fontSize: "11px" }}
                formatter={(val: any) => [`${Number(val).toFixed(1)} µs`, "Latency"]}
              />
              <Bar dataKey="latency" radius={[0, 4, 4, 0]}>
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Bandwidth Saturation Meter */}
      <div className="pt-2 border-t border-[#1f293d]">
        <div className="flex items-center justify-between mb-1 text-[11px]">
          <span className="text-slate-400 flex items-center gap-1.5 font-semibold">
            <Zap className="w-3.5 h-3.5 text-cyan-400" /> MEMORY BUS SATURATION
          </span>
          <span className="text-cyan-400 font-bold">
            {benchmark.achieved_bandwidth_gbs} GB/s ({benchmark.bandwidth_saturation_pct}%)
          </span>
        </div>
        <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
          <div
            className="h-full bg-gradient-to-r from-cyan-500 via-emerald-400 to-emerald-300 rounded-full transition-all duration-1000 shadow-sm shadow-cyan-500"
            style={{ width: `${Math.min(benchmark.bandwidth_saturation_pct, 100)}%` }}
          />
        </div>
        <div className="flex justify-between text-[10px] text-slate-500 mt-1">
          <span>0 GB/s</span>
          <span>Target A10G Peak: 600 GB/s</span>
        </div>
      </div>

      {/* Hardware Tiling Spec Callout */}
      {tilingPlan && (
        <div className="pt-2 border-t border-[#1f293d] grid grid-cols-3 gap-2 text-[11px] text-center text-slate-300">
          <div className="bg-[#131a29] p-1.5 rounded border border-[#212b40]">
            <span className="text-slate-500 block text-[9px]">BLOCK SIZE</span>
            <span className="text-cyan-400 font-bold">{tilingPlan.BLOCK_SIZE_M}</span>
          </div>
          <div className="bg-[#131a29] p-1.5 rounded border border-[#212b40]">
            <span className="text-slate-500 block text-[9px]">WARP COUNT</span>
            <span className="text-emerald-400 font-bold">{tilingPlan.num_warps} Warps</span>
          </div>
          <div className="bg-[#131a29] p-1.5 rounded border border-[#212b40]">
            <span className="text-slate-500 block text-[9px]">PIPELINE</span>
            <span className="text-amber-400 font-bold">{tilingPlan.num_stages} Stages</span>
          </div>
        </div>
      )}
    </div>
  );
};
