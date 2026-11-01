"use client";

import React from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Zap, Activity, CheckCircle, ShieldAlert } from "lucide-react";

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
      <div className="h-full flex flex-col items-center justify-center p-6 text-slate-500 border border-[#1f293d] rounded-lg bg-[#0e131d]/60">
        <Activity className="w-10 h-10 mb-3 text-slate-600 animate-pulse" />
        <span className="text-sm font-mono">Awaiting Kernel Dispatch...</span>
        <span className="text-xs text-slate-600 mt-1">Microsecond benchmark & roofline will render here</span>
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
    <div className="h-full flex flex-col gap-4 p-4 border border-[#1f293d] rounded-lg bg-[#0e131d]/90 font-mono text-xs">
      {/* Top Banner: Verification Badge & Speedup Highlights */}
      <div className="flex items-center justify-between pb-3 border-b border-[#1f293d]">
        <div className="flex items-center gap-2">
          {benchmark.numerical_parity_verified ? (
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-xs font-bold">
              <CheckCircle className="w-3.5 h-3.5" /> NUMERICAL PARITY: PASS
            </span>
          ) : (
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-rose-500/10 text-rose-400 border border-rose-500/30 text-xs font-bold">
              <ShieldAlert className="w-3.5 h-3.5" /> NUMERICAL MISMATCH
            </span>
          )}
        </div>
        <div className="text-right">
          <span className="text-emerald-400 font-bold text-sm">+{benchmark.speedup_vs_eager}x</span>
          <span className="text-slate-500 text-[11px] ml-1.5">vs Eager</span>
          <span className="text-slate-400 font-semibold text-xs ml-3">+{benchmark.speedup_vs_compile}x</span>
          <span className="text-slate-500 text-[11px] ml-1">vs Compile</span>
        </div>
      </div>

      {/* Latency Bar Chart */}
      <div className="flex-1 flex flex-col justify-center min-h-[160px]">
        <div className="text-[11px] text-slate-400 mb-2 flex items-center justify-between">
          <span>KERNEL EXECUTION LATENCY (LOWER IS BETTER)</span>
          <span className="text-slate-500">do_bench() µs</span>
        </div>
        <div className="h-[140px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 70, bottom: 5 }}>
              <XAxis type="number" stroke="#475569" tick={{ fontSize: 10 }} unit="µs" />
              <YAxis type="category" dataKey="name" stroke="#cbd5e1" tick={{ fontSize: 11 }} />
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
      <div className="pt-3 border-t border-[#1f293d]">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-[11px] text-slate-400 flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5 text-cyan-400" /> MEMORY BUS SATURATION
          </span>
          <span className="text-cyan-400 font-bold">
            {benchmark.achieved_bandwidth_gbs} GB/s ({benchmark.bandwidth_saturation_pct}%)
          </span>
        </div>
        <div className="w-full h-2.5 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
          <div
            className="h-full bg-gradient-to-r from-cyan-500 to-emerald-400 rounded-full transition-all duration-1000"
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
            <span className="text-slate-500 block text-[9px]">PIPELINE STAGES</span>
            <span className="text-amber-400 font-bold">{tilingPlan.num_stages} Stages</span>
          </div>
        </div>
      )}
    </div>
  );
};
