import { useMemo } from 'react';
import { HeartRateData } from '../lib/supabase';
import { Activity } from 'lucide-react';

interface HeartRateChartProps {
  data: HeartRateData[];
}

export function HeartRateChart({ data }: HeartRateChartProps) {
  const chartData = useMemo(() => {
    if (!data.length) return null;

    const minHR = Math.min(...data.map(d => d.heart_rate_bpm));
    const maxHR = Math.max(...data.map(d => d.heart_rate_bpm));
    const avgHR = data.reduce((sum, d) => sum + d.heart_rate_bpm, 0) / data.length;

    const padding = 10;
    const rangeMin = Math.max(40, minHR - padding);
    const rangeMax = Math.min(140, maxHR + padding);

    const points = data.map((point, index) => {
      const x = (index / (data.length - 1)) * 100;
      const normalizedY = ((point.heart_rate_bpm - rangeMin) / (rangeMax - rangeMin));
      const y = 100 - (normalizedY * 100);
      return { x, y, hr: point.heart_rate_bpm, time: point.timestamp_ms };
    });

    return {
      points,
      minHR: Math.round(minHR),
      maxHR: Math.round(maxHR),
      avgHR: Math.round(avgHR * 10) / 10,
      rangeMin,
      rangeMax,
    };
  }, [data]);

  if (!chartData) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6 text-center text-gray-500">
        <Activity className="w-12 h-12 mx-auto mb-2 opacity-50" />
        <p>No heart rate data available</p>
      </div>
    );
  }

  const pathD = chartData.points
    .map((point, index) => {
      const command = index === 0 ? 'M' : 'L';
      return `${command} ${point.x} ${point.y}`;
    })
    .join(' ');

  const areaPathD = `${pathD} L 100 100 L 0 100 Z`;

  const formatTime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Activity className="w-6 h-6 text-red-600" />
          <h3 className="text-xl font-semibold text-gray-900">Heart Rate Analysis</h3>
        </div>
        <div className="flex items-center space-x-6 text-sm">
          <div className="text-center">
            <div className="text-gray-500">Average</div>
            <div className="text-2xl font-bold text-gray-900">{chartData.avgHR}</div>
            <div className="text-gray-400 text-xs">BPM</div>
          </div>
          <div className="text-center">
            <div className="text-gray-500">Min</div>
            <div className="text-xl font-semibold text-blue-600">{chartData.minHR}</div>
            <div className="text-gray-400 text-xs">BPM</div>
          </div>
          <div className="text-center">
            <div className="text-gray-500">Max</div>
            <div className="text-xl font-semibold text-red-600">{chartData.maxHR}</div>
            <div className="text-gray-400 text-xs">BPM</div>
          </div>
        </div>
      </div>

      <div className="relative">
        <svg
          viewBox="0 0 100 100"
          className="w-full h-64"
          preserveAspectRatio="none"
        >
          <defs>
            <linearGradient id="heartRateGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="rgb(239, 68, 68)" stopOpacity="0.3" />
              <stop offset="100%" stopColor="rgb(239, 68, 68)" stopOpacity="0.05" />
            </linearGradient>
          </defs>

          <path
            d={areaPathD}
            fill="url(#heartRateGradient)"
          />

          <path
            d={pathD}
            fill="none"
            stroke="rgb(239, 68, 68)"
            strokeWidth="0.5"
            vectorEffect="non-scaling-stroke"
          />

          {chartData.points.map((point, index) => (
            <g key={index}>
              <circle
                cx={point.x}
                cy={point.y}
                r="0.8"
                fill="rgb(220, 38, 38)"
                className="cursor-pointer hover:r-1.5 transition-all"
              >
                <title>{`${formatTime(point.time)}: ${Math.round(point.hr)} BPM`}</title>
              </circle>
            </g>
          ))}

          {[0, 25, 50, 75, 100].map((y) => (
            <line
              key={y}
              x1="0"
              y1={y}
              x2="100"
              y2={y}
              stroke="rgb(229, 231, 235)"
              strokeWidth="0.2"
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>

        <div className="absolute left-0 top-0 bottom-0 w-12 flex flex-col justify-between text-xs text-gray-500 -ml-12">
          <div>{chartData.rangeMax}</div>
          <div>{Math.round((chartData.rangeMax + chartData.rangeMin) / 2)}</div>
          <div>{chartData.rangeMin}</div>
        </div>
      </div>

      <div className="flex justify-between text-xs text-gray-500 mt-4">
        <div>0:00</div>
        <div>Time</div>
        <div>{formatTime(chartData.points[chartData.points.length - 1]?.time || 0)}</div>
      </div>

      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-semibold text-gray-700 mb-2">Heart Rate Variability</h4>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Range:</span>
            <span className="ml-2 font-semibold">{chartData.maxHR - chartData.minHR} BPM</span>
          </div>
          <div>
            <span className="text-gray-600">Data Points:</span>
            <span className="ml-2 font-semibold">{data.length}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
