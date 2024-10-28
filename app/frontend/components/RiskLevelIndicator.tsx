import React from 'react';
import { AlertCircle, Info } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

const RiskLevelIndicator = ({ confidence, result } : { confidence : any, result : any}) => {
  // Determine risk level based on result and confidence
  const getRiskLevel = () => {
    if (result === "Malignant") {
      if (confidence > 0.8) return "High Risk";
      if (confidence > 0.6) return "Moderate-High Risk";
      return "Moderate Risk";
    } else {
      if (confidence > 0.8) return "Low Risk";
      if (confidence > 0.6) return "Low-Moderate Risk";
      return "Moderate Risk";
    }
  };

  // Get explanation based on risk level
  const getExplanation = () => {
    const riskLevel = getRiskLevel();
    const explanations = {
      "High Risk": "Immediate medical consultation is strongly recommended. The analysis shows strong indicators of malignancy.",
      "Moderate-High Risk": "Prompt medical evaluation is recommended. While not conclusive, there are significant concerning indicators.",
      "Moderate Risk": "Medical evaluation is recommended. The results show some concerning indicators but are not conclusive.",
      "Low-Moderate Risk": "Follow-up with healthcare provider is advised. While indicators are mostly benign, monitoring is recommended.",
      "Low Risk": "Regular screening schedule can be maintained. The indicators strongly suggest benign characteristics."
    };
    return explanations[riskLevel];
  };

  // Get color based on risk level
  const getColorClass = () => {
    const riskLevel = getRiskLevel();
    const colors = {
      "High Risk": "bg-red-500/20 text-red-400",
      "Moderate-High Risk": "bg-orange-500/20 text-orange-400",
      "Moderate Risk": "bg-yellow-500/20 text-yellow-400",
      "Low-Moderate Risk": "bg-blue-500/20 text-blue-400",
      "Low Risk": "bg-green-500/20 text-green-400"
    };
    return colors[riskLevel];
  };

  return (
    <Alert className={`mt-4 ${getColorClass()}`}>
      <AlertCircle className="h-4 w-4" />
      <AlertTitle className="flex items-center gap-2">
        Risk Level: {getRiskLevel()}
        <span className="text-sm font-normal">
          ({(confidence * 100).toFixed(1)}% confidence)
        </span>
      </AlertTitle>
      <AlertDescription className="mt-2">
        {getExplanation()}
      </AlertDescription>
    </Alert>
  );
};

export default RiskLevelIndicator;