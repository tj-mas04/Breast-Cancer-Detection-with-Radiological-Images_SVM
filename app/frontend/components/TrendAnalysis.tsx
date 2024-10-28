import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const TrendAnalysis = ({ screenings } : { screenings : any }) => {
  // Process screenings data for visualization
  const processData = () => {
    return screenings.map(screening  => ({
      date: new Date(screening.created_at).toLocaleDateString(),
      confidence: parseFloat((screening.confidence * 100).toFixed(1)),
      result: screening.result
    })).slice(-10); // Show last 10 screenings
  };

  return (
    <div className="w-full my-6 p-4 bg-slate-800 rounded-lg">
      <h3 className="text-xl font-bold mb-4 text-slate-100">Analysis Trends</h3>
      <div className="w-full h-[300px]">
        <LineChart data={processData()} width={600} height={300}>
          <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
          <XAxis 
            dataKey="date" 
            stroke="#94a3b8"
            tick={{ fill: '#94a3b8' }}
          />
          <YAxis 
            stroke="#94a3b8"
            tick={{ fill: '#94a3b8' }}
            label={{ 
              value: 'Confidence %', 
              angle: -90, 
              position: 'insideLeft',
              fill: '#94a3b8'
            }} 
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#1e293b',
              border: '1px solid #475569',
              borderRadius: '6px',
              color: '#e2e8f0'
            }}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="confidence" 
            stroke="#38bdf8" 
            strokeWidth={2}
            dot={{ fill: '#38bdf8' }}
          />
        </LineChart>
      </div>
    </div>
  );
};

export default TrendAnalysis;