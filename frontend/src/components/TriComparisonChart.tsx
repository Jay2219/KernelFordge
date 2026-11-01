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

export const TriComparisonChart: React.FC<TriComparisonChartProps> = ({ benchmark, tilingPlan }) => {
  if (!benchmark) {
    return (
      <div className="h-full flex flex-col justify-between p-4 border border-[#1f293d] rounded-lg bg-[#0e131d]/90 font-mono text-xs">
        {/* Top Banner Skeleton */}
        <div className="flex items-center justify-between pb-3 border-b border-[#1f293d]/60">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-900/80 border border-slate-800 text-slate-500 text-[11px] font-semibold">
            <Activity className="w-3.5 h-3.5 text-slate-600 animate-pulse" />
            <span>STANDBY // ROOFLINE READY</span>
          </div>
          <div className="text-right text-slate-600 text-[11px]">
            <span>+--x vs Eager</span>
            <span className="mx-2 text-slate-700">|</span>
            <span>+--x vs Compile</span>
          </div>
        </div>

        {/* Ghosted Latency Bars */}
        <div className="flex flex-col gap-3 my-2">
          <div className="text-[11px] text-slate-500 flex items-center justify-between">
            <span>KERNEL EXECUTION LATENCY</span>
            <span className="text-slate-600">do_bench() µs</span>
          </div>

          {/* Bar 1: PyTorch Eager Skeleton */}
          <div className="space-y-1">
            <div className="flex justify-between text-[10px] text-slate-500">
              <span className="text-rose-400/60 font-semibold">PyTorch Eager</span>
              <span>--.- µs</span>
            </div>
            <div className="w-full h-5 bg-[#121722] rounded border border-rose-950/40 overflow-hidden relative">
              <div className="w-[85%] h-full bg-rose-500/10 border-r border-rose-500/30" />
            </div>
          </div>

          {/* Bar 2: torch.compile Skeleton */}
          <div className="space-y-1">
            <div className="flex justify-between text-[10px] text-slate-500">
              <span className="text-amber-400/60 font-semibold">torch.compile (Inductor)</span>
              <span>--.- µs</span>
            </div>
            <div className="w-full h-5 bg-[#121722] rounded border border-amber-950/40 overflow-hidden relative">
              <div className="w-[45%] h-full bg-amber-500/10 border-r border-amber-500/30" />
            </div>
          </div>

          {/* Bar 3: KernelForge Triton Skeleton */}
          <div className="space-y-1">
            <div className="flex justify-between text-[10px] text-slate-500">
              <span className="text-emerald-400/70 font-semibold">KernelForge Triton</span>
              <span className="text-emerald-500/70">TARGET: &lt;120 µs</span>
            </div>
            <div className="w-full h-5 bg-[#121722] rounded border border-emerald-950/40 overflow-hidden relative">
              <div className="w-[30%] h-full bg-emerald-500/15 border-r border-emerald-400/40" />
            </div>
          </div>
        </div>

        {/* Ghosted Bandwidth Saturation Meter */}
        <div className="pt-3 border-t border-[#1f293d]/60">
          <div className="flex items-center justify-between mb-1.5 text-[11px]">
            <span className="text-slate-500 flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-slate-600" /> MEMORY BUS SATURATION
            </span>
            <span className="text-slate-500">--.- GB/s (--%)</span>
          </div>
          <div className="w-full h-2.5 bg-[#121722] rounded-full overflow-hidden border border-slate-800">
            <div className="w-[88%] h-full bg-cyan-500/10 border-r border-cyan-500/20" />
          </div>
          <div className="flex justify-between text-[10px] text-slate-600 mt-1">
            <span>0 GB/s</span>
            <span>Target A10G Peak: 600 GB/s</span>
          </div>
        </div>

        {/* Ghosted Hardware Tiling Grid */}
        <div className="pt-2 border-t border-[#1f293d]/60 grid grid-cols-3 gap-2 text-[11px] text-center text-slate-500">
          <div className="bg-[#121724]/80 p-1.5 rounded border border-slate-800/80">
            <span className="text-slate-600 block text-[9px]">BLOCK SIZE</span>
            <span className="text-slate-400 font-bold">1024</span>
          </div>
          <div className="bg-[#121724]/80 p-1.5 rounded border border-slate-800/80">
            <span className="text-slate-600 block text-[9px]">WARP COUNT</span>
            <span className="text-slate-400 font-bold">4 Warps</span>
          </div>
          <div className="bg-[#121724]/80 p-1.5 rounded border border-slate-800/80">
            <span className="text-slate-600 block text-[9px]">STAGES</span>
            <span className="text-slate-400 font-bold">3 Stages</span>
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
      <div className="flex items-center justify-between pb-3 border-b border-[#1f293d]">
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
          <span className="text-emerald-400 font-bold text-sm">+{benchmark.speedup_vs_eager}x</span>
          <span className="text-slate-500 text-[10px] ml-1">vs Eager</span>
          <span className="text-cyan-300 font-semibold text-xs ml-2.5">+{benchmark.speedup_vs_compile}x</span>
          <span className="text-slate-500 text-[10px] ml-1">vs Compile</span>
        </div>
      </div>

      {/* Latency Bar Chart */}
      <div className="flex flex-col justify-center my-1">
        <div className="text-[11px] text-slate-400 mb-1 flex items-center justify-between">
          <span>KERNEL LATENCY COMPARISON</span>
          <span className="text-slate-500">do_bench() µs</span>
        </div>
        <div className="h-[135px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 35, left: 60, bottom: 5 }}>
              <XAxis type="number" stroke="#475569" tick={{ fontSize: 10 }} unit="µs" />
              <YAxis type="category" dataKey="name" stroke="#cbd5e1" tick={{ fontSize: 10 }} width={75} />
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
          <span className="text-slate-400 flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5 text-cyan-400" /> MEMORY BUS SATURATION
          </span>
          <span className="text-cyan-400 font-bold">
            {benchmark.achieved_bandwidth_gbs} GB/s ({benchmark.bandwidth_saturation_pct}%)
          </span>
        </div>
        <div className="w-full h-2.5 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
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
