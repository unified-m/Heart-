import { RiskPrediction } from '../lib/supabase';
import { AlertTriangle, Shield, AlertCircle, TrendingUp, Lightbulb } from 'lucide-react';

interface RiskAssessmentProps {
  prediction: RiskPrediction;
}

export function RiskAssessment({ prediction }: RiskAssessmentProps) {
  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low':
        return {
          bg: 'bg-green-50',
          border: 'border-green-200',
          text: 'text-green-800',
          icon: 'text-green-600',
          badge: 'bg-green-100',
        };
      case 'medium':
        return {
          bg: 'bg-yellow-50',
          border: 'border-yellow-200',
          text: 'text-yellow-800',
          icon: 'text-yellow-600',
          badge: 'bg-yellow-100',
        };
      case 'high':
        return {
          bg: 'bg-red-50',
          border: 'border-red-200',
          text: 'text-red-800',
          icon: 'text-red-600',
          badge: 'bg-red-100',
        };
      default:
        return {
          bg: 'bg-gray-50',
          border: 'border-gray-200',
          text: 'text-gray-800',
          icon: 'text-gray-600',
          badge: 'bg-gray-100',
        };
    }
  };

  const getRiskIcon = (level: string) => {
    switch (level) {
      case 'low':
        return Shield;
      case 'medium':
        return AlertCircle;
      case 'high':
        return AlertTriangle;
      default:
        return Shield;
    }
  };

  const colors = getRiskColor(prediction.risk_level);
  const RiskIcon = getRiskIcon(prediction.risk_level);

  const getRiskLabel = (level: string) => {
    return level.charAt(0).toUpperCase() + level.slice(1) + ' Risk';
  };

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      <div className={`${colors.bg} border-b ${colors.border} p-6`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className={`p-3 rounded-full ${colors.badge}`}>
              <RiskIcon className={`w-8 h-8 ${colors.icon}`} />
            </div>
            <div>
              <h3 className="text-2xl font-bold text-gray-900">Risk Assessment</h3>
              <p className={`text-sm ${colors.text} font-semibold mt-1`}>
                {getRiskLabel(prediction.risk_level)}
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-gray-500 text-sm mb-1">Risk Score</div>
            <div className={`text-4xl font-bold ${colors.text}`}>
              {prediction.risk_score}
            </div>
            <div className="text-gray-400 text-xs">out of 100</div>
          </div>
        </div>

        <div className="mt-4">
          <div className="w-full bg-white rounded-full h-3 overflow-hidden">
            <div
              className={`h-full ${prediction.risk_level === 'low' ? 'bg-green-500' : prediction.risk_level === 'medium' ? 'bg-yellow-500' : 'bg-red-500'} transition-all duration-500`}
              style={{ width: `${prediction.risk_score}%` }}
            />
          </div>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {prediction.insights.variability && (
          <div>
            <div className="flex items-center space-x-2 mb-2">
              <TrendingUp className="w-5 h-5 text-blue-600" />
              <h4 className="font-semibold text-gray-900">Heart Rate Variability</h4>
            </div>
            <p className="text-gray-700 text-sm pl-7">{prediction.insights.variability}</p>
          </div>
        )}

        {prediction.insights.trend && (
          <div>
            <div className="flex items-center space-x-2 mb-2">
              <TrendingUp className="w-5 h-5 text-blue-600" />
              <h4 className="font-semibold text-gray-900">Trend Analysis</h4>
            </div>
            <p className="text-gray-700 text-sm pl-7">{prediction.insights.trend}</p>
          </div>
        )}

        {prediction.insights.anomalies && prediction.insights.anomalies.length > 0 && (
          <div>
            <div className="flex items-center space-x-2 mb-3">
              <AlertCircle className="w-5 h-5 text-orange-600" />
              <h4 className="font-semibold text-gray-900">Detected Anomalies</h4>
            </div>
            <ul className="space-y-2 pl-7">
              {prediction.insights.anomalies.map((anomaly, index) => (
                <li key={index} className="flex items-start space-x-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-orange-500 mt-2 flex-shrink-0" />
                  <span className="text-gray-700 text-sm">{anomaly}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {prediction.insights.recommendations && prediction.insights.recommendations.length > 0 && (
          <div className={`${colors.bg} border ${colors.border} rounded-lg p-4`}>
            <div className="flex items-center space-x-2 mb-3">
              <Lightbulb className={`w-5 h-5 ${colors.icon}`} />
              <h4 className={`font-semibold ${colors.text}`}>Recommendations</h4>
            </div>
            <ul className="space-y-2">
              {prediction.insights.recommendations.map((recommendation, index) => (
                <li key={index} className="flex items-start space-x-2">
                  <span className={`w-1.5 h-1.5 rounded-full ${prediction.risk_level === 'low' ? 'bg-green-500' : prediction.risk_level === 'medium' ? 'bg-yellow-500' : 'bg-red-500'} mt-2 flex-shrink-0`} />
                  <span className={`text-sm ${colors.text}`}>{recommendation}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="pt-4 border-t border-gray-200">
          <p className="text-xs text-gray-500">
            Analysis generated on {new Date(prediction.predicted_at).toLocaleString()}
          </p>
          <p className="text-xs text-gray-400 mt-1">
            This assessment is for informational purposes only and should not replace professional medical advice.
          </p>
        </div>
      </div>
    </div>
  );
}
