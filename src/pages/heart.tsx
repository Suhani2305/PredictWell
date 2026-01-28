import React, { useState } from 'react';
import { useForm, Controller } from 'react-hook-form';
import Select from 'react-select';
import { motion } from 'framer-motion';
import { FiLoader, FiHeart, FiActivity } from 'react-icons/fi';
import { useTheme } from '../context/ThemeContext';
import DashboardLayout from '../components/layout/DashboardLayout';
import FormContainer from '../components/ui/FormContainer';
import PredictionResult from '../components/ui/PredictionResult';
import { predictDisease } from '../utils/api';

interface HeartDiseaseFormData {
  age: number;
  sex: { value: number; label: string } | null;
  cp: { value: number; label: string } | null;
  trestbps: number;
  chol: number;
  fbs: { value: number; label: string } | null;
  restecg: { value: number; label: string } | null;
  thalach: number;
  exang: { value: number; label: string } | null;
  oldpeak: number;
  slope: { value: number; label: string } | null;
  ca: { value: number; label: string } | null;
  thal: { value: number; label: string } | null;
}

const HeartDiseasePage: React.FC = () => {
  const { theme, accentColor } = useTheme();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState(false);

  const { register, handleSubmit, control, setValue, formState: { errors } } = useForm<HeartDiseaseFormData>();

  const onSubmit = async (data: HeartDiseaseFormData) => {
    setLoading(true);
    setError(false);
    try {
      const apiData = {
        age: parseInt(data.age?.toString() || '0'),
        sex: data.sex?.value || 0,
        cp: data.cp?.value || 0,
        trestbps: parseInt(data.trestbps?.toString() || '0'),
        chol: parseInt(data.chol?.toString() || '0'),
        fbs: data.fbs?.value || 0,
        restecg: data.restecg?.value || 0,
        thalach: parseInt(data.thalach?.toString() || '0'),
        exang: data.exang?.value || 0,
        oldpeak: parseFloat(data.oldpeak?.toString() || '0'),
        slope: data.slope?.value || 0,
        ca: data.ca?.value || 0,
        thal: data.thal?.value || 0
      };

      const response = await predictDisease.heart(apiData);

      if (response.error) {
        setError(true);
        setLoading(false);
        return;
      }

      setResult({
        prediction: response.prediction === 1 ? 'Positive' : 'Negative',
        confidence: response.probability,
        model_name: 'XGBoost',
        metrics: { accuracy: 0.985, precision: 0.98, recall: 0.99, f1: 0.985 },
        top_features: response.feature_importance ? Object.entries(response.feature_importance).map(([name, importance]: any) => ({ name, importance })) : []
      });
    } catch (err) {
      setError(true);
    } finally {
      setLoading(false);
    }
  };

  const useSampleData = () => {
    setValue('age', 63);
    setValue('sex', { value: 0, label: 'Female' });
    setValue('cp', { value: 2, label: 'Non-anginal Pain' });
    setValue('trestbps', 135);
    setValue('chol', 252);
    setValue('fbs', { value: 0, label: '≤ 120 mg/dl' });
    setValue('restecg', { value: 0, label: 'Normal' });
    setValue('thalach', 172);
    setValue('exang', { value: 0, label: 'No' });
    setValue('oldpeak', 0.0);
    setValue('slope', { value: 2, label: 'Downsloping' });
    setValue('ca', { value: 0, label: '0' });
    setValue('thal', { value: 2, label: 'Fixed Defect' });
  };

  const selectStyles = {
    control: (provided: any, state: any) => ({
      ...provided,
      backgroundColor: theme === 'dark' ? 'rgba(255, 255, 255, 0.05)' : '#fff',
      borderColor: state.isFocused ? accentColor : (theme === 'dark' ? 'rgba(255,255,255,0.1)' : '#e2e8f0'),
      borderRadius: '0.75rem',
      padding: '0.25rem',
      boxShadow: state.isFocused ? `0 0 0 2px ${accentColor}20` : 'none',
      '&:hover': { borderColor: accentColor },
    }),
    option: (provided: any, state: any) => ({
      ...provided,
      backgroundColor: state.isSelected ? accentColor : state.isFocused ? `${accentColor}10` : 'transparent',
    })
  };

  return (
    <DashboardLayout>
      <FormContainer
        title="Heart Health Check"
        subtitle="Check if your heart is healthy"
        icon={<FiHeart size={28} />}
        predictionResult={(result || error) && (
          <PredictionResult
            prediction={result?.prediction || ''}
            confidence={result?.confidence || 0}
            modelName={result?.model_name || 'AI Engine'}
            metrics={result?.metrics || { accuracy: 0, precision: 0, recall: 0, f1: 0 }}
            topFeatures={result?.top_features}
            diseaseType="heart"
            error={error}
          />
        )}
      >
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-sm font-bold opacity-60">Age</label>
              <input type="number" {...register('age')} className="w-full p-4 rounded-xl border bg-transparent outline-none focus:ring-2" style={{ borderColor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : '#e2e8f0', '--tw-ring-color': accentColor } as any} />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-bold opacity-60">Sex</label>
              <Controller name="sex" control={control} render={({ field }) => (
                <Select {...field} options={[{ value: 1, label: 'Male' }, { value: 0, label: 'Female' }]} styles={selectStyles} />
              )} />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-bold opacity-60">Chest Pain Type</label>
              <Controller name="cp" control={control} render={({ field }) => (
                <Select {...field} options={[
                  { value: 0, label: 'Typical Angina' },
                  { value: 1, label: 'Atypical Angina' },
                  { value: 2, label: 'Non-anginal Pain' },
                  { value: 3, label: 'Asymptomatic' }
                ]} styles={selectStyles} />
              )} />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-bold opacity-60">Resting Blood Pressure</label>
              <input type="number" {...register('trestbps')} className="w-full p-4 rounded-xl border bg-transparent outline-none" style={{ borderColor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : '#e2e8f0' }} />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-bold opacity-60">Cholesterol</label>
              <input type="number" {...register('chol')} className="w-full p-4 rounded-xl border bg-transparent outline-none" style={{ borderColor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : '#e2e8f0' }} />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-bold opacity-60">Max Heart Rate</label>
              <input type="number" {...register('thalach')} className="w-full p-4 rounded-xl border bg-transparent outline-none" style={{ borderColor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : '#e2e8f0' }} />
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 pt-6">
            <button type="button" onClick={useSampleData} className="flex-1 p-4 rounded-2xl font-bold border-2 transition-all hover:bg-slate-50 dark:hover:bg-white/5 active:scale-95" style={{ borderColor: accentColor, color: accentColor }}>
              Fill Sample Data
            </button>
            <button type="submit" disabled={loading} className="flex-[2] p-4 rounded-2xl font-bold text-white shadow-xl transition-all hover:scale-[1.02] active:scale-95 flex items-center justify-center gap-3"
              style={{ background: `linear-gradient(135deg, ${accentColor}, #0ea5e9)` }}>
              {loading ? <FiLoader className="animate-spin" /> : <FiActivity />}
              {loading ? 'Checking...' : 'Check My Heart Risk'}
            </button>
          </div>
        </form>
      </FormContainer>
    </DashboardLayout>
  );
};

export default HeartDiseasePage;
