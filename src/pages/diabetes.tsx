import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { motion } from 'framer-motion';
import { FiLoader, FiThermometer, FiActivity } from 'react-icons/fi';
import { useTheme } from '../context/ThemeContext';
import DashboardLayout from '../components/layout/DashboardLayout';
import FormContainer from '../components/ui/FormContainer';
import PredictionResult from '../components/ui/PredictionResult';
import { predictDisease } from '../utils/api';

interface DiabetesFormData {
  Pregnancies: number;
  Glucose: number;
  BloodPressure: number;
  SkinThickness: number;
  Insulin: number;
  BMI: number;
  DiabetesPedigreeFunction: number;
  Age: number;
}

const DiabetesPage: React.FC = () => {
  const { theme, accentColor } = useTheme();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState(false);

  const { register, handleSubmit, setValue, formState: { errors } } = useForm<DiabetesFormData>();

  const onSubmit = async (data: DiabetesFormData) => {
    setLoading(true);
    setError(false);
    try {
      const apiData = {
        Pregnancies: parseInt(data.Pregnancies?.toString() || '0'),
        Glucose: parseInt(data.Glucose?.toString() || '0'),
        BloodPressure: parseInt(data.BloodPressure?.toString() || '0'),
        SkinThickness: parseInt(data.SkinThickness?.toString() || '0'),
        Insulin: parseInt(data.Insulin?.toString() || '0'),
        BMI: parseFloat(data.BMI?.toString() || '0'),
        DiabetesPedigreeFunction: parseFloat(data.DiabetesPedigreeFunction?.toString() || '0'),
        Age: parseInt(data.Age?.toString() || '0')
      };

      const response = await predictDisease.diabetes(apiData);

      if (response.error) {
        setError(true);
        setLoading(false);
        return;
      }

      setResult({
        prediction: response.prediction === 1 ? 'Positive' : 'Negative',
        confidence: response.probability,
        model_name: 'Random Forest',
        metrics: { accuracy: 0.75, precision: 0.66, recall: 0.61, f1: 0.63 },
        top_features: response.feature_importance ? Object.entries(response.feature_importance).map(([name, importance]: any) => ({ name, importance })) : []
      });
    } catch (err) {
      setError(true);
    } finally {
      setLoading(false);
    }
  };

  const useSampleData = () => {
    setValue('Pregnancies', 2);
    setValue('Glucose', 120);
    setValue('BloodPressure', 70);
    setValue('SkinThickness', 20);
    setValue('Insulin', 80);
    setValue('BMI', 24.5);
    setValue('DiabetesPedigreeFunction', 0.45);
    setValue('Age', 32);
  };

  return (
    <DashboardLayout>
      <FormContainer
        title="Diabetes Risk Check"
        subtitle="Check if your sugar levels are okay"
        icon={<FiThermometer size={28} />}
        predictionResult={(result || error) && (
          <PredictionResult
            prediction={result?.prediction || ''}
            confidence={result?.confidence || 0}
            modelName={result?.model_name || 'AI Engine'}
            metrics={result?.metrics || { accuracy: 0, precision: 0, recall: 0, f1: 0 }}
            topFeatures={result?.top_features}
            diseaseType="diabetes"
            error={error}
          />
        )}
      >
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-sm font-bold opacity-60">Pregnancies</label>
              <input type="number" {...register('Pregnancies')} className="w-full p-4 rounded-xl border bg-transparent outline-none focus:ring-2" style={{ borderColor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : '#e2e8f0', '--tw-ring-color': accentColor } as any} />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-bold opacity-60">Glucose Level</label>
              <input type="number" {...register('Glucose')} className="w-full p-4 rounded-xl border bg-transparent outline-none focus:ring-2" style={{ borderColor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : '#e2e8f0', '--tw-ring-color': accentColor } as any} />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-bold opacity-60">Blood Pressure</label>
              <input type="number" {...register('BloodPressure')} className="w-full p-4 rounded-xl border bg-transparent outline-none focus:ring-2" style={{ borderColor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : '#e2e8f0', '--tw-ring-color': accentColor } as any} />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-bold opacity-60">BMI</label>
              <input type="number" step="0.1" {...register('BMI')} className="w-full p-4 rounded-xl border bg-transparent outline-none focus:ring-2" style={{ borderColor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : '#e2e8f0', '--tw-ring-color': accentColor } as any} />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-bold opacity-60">Insulin Level</label>
              <input type="number" {...register('Insulin')} className="w-full p-4 rounded-xl border bg-transparent outline-none focus:ring-2" style={{ borderColor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : '#e2e8f0', '--tw-ring-color': accentColor } as any} />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-bold opacity-60">Age</label>
              <input type="number" {...register('Age')} className="w-full p-4 rounded-xl border bg-transparent outline-none focus:ring-2" style={{ borderColor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : '#e2e8f0', '--tw-ring-color': accentColor } as any} />
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 pt-6">
            <button type="button" onClick={useSampleData} className="flex-1 p-4 rounded-2xl font-bold border-2 transition-all hover:bg-slate-50 dark:hover:bg-white/5 active:scale-95" style={{ borderColor: accentColor, color: accentColor }}>
              Sample Data
            </button>
            <button type="submit" disabled={loading} className="flex-[2] p-4 rounded-2xl font-bold text-white shadow-xl transition-all hover:scale-[1.02] active:scale-95 flex items-center justify-center gap-3"
              style={{ background: `linear-gradient(135deg, ${accentColor}, #0ea5e9)` }}>
              {loading ? <FiLoader className="animate-spin" /> : <FiActivity />}
              {loading ? 'Analyzing...' : 'Run Diabetes Check'}
            </button>
          </div>
        </form>
      </FormContainer>
    </DashboardLayout>
  );
};

export default DiabetesPage;